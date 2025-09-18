/* global_done_tree.c
 *
 * Tree-based global termination for OpenSHMEM.
 * - Leaf groups of size G (ENV: GLOBAL_GROUP_SIZE, default 8).
 * - "Last PE in a group" detection flips a done flag at the group's leader PE.
 * - Leaders propagate up a binary tree of groups via parent flags.
 * - Root (PE 0) prints aggregated elapsed times and coordinates a two-phase exit
 *   so that non-roots exit first; root exits last to avoid RMA-after-teardown.
 *
 * Debug:
 *   GLOBAL_DONE_DEBUG=0 (default) -> quiet aggregate only
 *   GLOBAL_DONE_DEBUG=1           -> verbose per-step prints
 *
 * Compile:
 *   oshcc -O3 -std=c11 -o global_done_tree global_done_tree.c
 *
 * Run:
 *   srun --mpi=pmix -N 1 -n 24 ./global_done_tree
 */

 #define _POSIX_C_SOURCE 199309L

 #include <shmem.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <time.h>
 
 /* ---------- timing ---------- */
 static inline double now_sec(void) {
     struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC, &ts);
     return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
 }
 
 /* ---------- globals / symmetric ---------- */
 static int    *LOCAL_DONE;      /* -1 = done, 0 = not done */
 static double *ELAPSED_MS;      /* per-PE elapsed time (ms) */
 static int    *AGG_PRINTED;     /* print-once flag at ROOT_PE */
 static int    *ROOT_GO;         /* on ROOT_PE: 0 = hold, 1 = non-roots may exit */
 
 static int   **GROUP_DONE;      /* per level array of group flags (symmetric) */
 static int     MAX_LEVELS;      /* number of levels including root level */
 static int    *NUM_GROUPS;      /* number of groups at each level */
 
 static int     G_LEAF = 8;      /* leaf group size (can be changed via env) */
 static int     g_debug = 0;
 static double  g_start_time = 0.0;
 static const int ROOT_PE = 0;
 
 /* ---------- helpers ---------- */
 
 static int env_debug_enabled(void) {
     const char *e = getenv("GLOBAL_DONE_DEBUG");
     if (!e) return 0;
     if (e[0] == '\0' || e[0] == '0') return 0;
     return 1;
 }
 
 static int env_group_size(void) {
     const char *e = getenv("GLOBAL_GROUP_SIZE");
     if (!e || e[0] == '\0') return 8;
     int v = atoi(e);
     return (v >= 1) ? v : 8;
 }
 
 static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
 
 /* size (in PEs) of a group at level L (L=0 is leaf) in a binary tree of groups */
 static inline int group_span_at_level(int leaf_size, int level) {
     return leaf_size << level; /* leaf_size * (2^level) */
 }
 
 /* number of groups at level L */
 static inline int num_groups_at_level(int npes, int leaf_size, int level) {
     int span = group_span_at_level(leaf_size, level);
     return ceil_div(npes, span);
 }
 
 /* group leader PE id for (level, group_idx) */
 static inline int group_leader_pe(int leaf_size, int level, int group_idx) {
     return group_idx * group_span_at_level(leaf_size, level);
 }
 
 /* child group indices given parent index */
 static inline int left_child_idx(int parent_idx)  { return parent_idx * 2; }
 static inline int right_child_idx(int parent_idx) { return parent_idx * 2 + 1; }
 
 /* ---------- allocation of symmetric flag arrays per level ---------- */
 static void allocate_tree_flags(int npes) {
     /* Determine number of levels until root (1 group) */
     int levels = 0;
     while (1) {
         int ng = num_groups_at_level(npes, G_LEAF, levels);
         levels++;
         if (ng <= 1) break;
     }
     MAX_LEVELS = levels; /* includes the top (root) level */
 
     NUM_GROUPS = shmem_malloc(sizeof(int) * MAX_LEVELS);
     GROUP_DONE = shmem_malloc(sizeof(int*) * MAX_LEVELS);
     if (!NUM_GROUPS || !GROUP_DONE) {
         shmem_global_exit(1);
     }
 
     for (int L = 0; L < MAX_LEVELS; L++) {
         int ng = num_groups_at_level(npes, G_LEAF, L);
         NUM_GROUPS[L] = ng;
         GROUP_DONE[L] = shmem_malloc(sizeof(int) * ng);
         if (!GROUP_DONE[L]) shmem_global_exit(1);
         for (int i = 0; i < ng; i++) GROUP_DONE[L][i] = 0; /* 0 = not done, 1 = done */
     }
 }
 
 /* Spin helper: small backoff to avoid hammering */
 static inline void tiny_pause(void) {
     struct timespec ts = {0, 1000000}; /* 1 ms */
     nanosleep(&ts, NULL);
 }
 
 /* ---------- root print + coordinated exit ---------- */
 
 static void root_print_then_release_and_exit(void) {
     int npes = shmem_n_pes();
     int me   = shmem_my_pe();
 
     /* Print aggregate ONCE (root only). */
     int old = shmem_int_atomic_compare_swap(AGG_PRINTED, 0, 1, ROOT_PE);
     if (old == 0 && me == ROOT_PE) {
         double sum = 0.0, minv = 0.0, maxv = 0.0, val;
         for (int pe = 0; pe < npes; pe++) {
             val = (pe == me) ? *ELAPSED_MS : shmem_double_g(ELAPSED_MS, pe);
             if (pe == 0) { minv = maxv = val; }
             if (val < minv) minv = val;
             if (val > maxv) maxv = val;
             sum += val;
         }
         double avg = sum / (double)npes;
         printf("Aggregated ELAPSED_MS across %d PEs: min=%.3f ms  avg=%.3f ms  max=%.3f ms\n",
                npes, minv, avg, maxv);
         fflush(stdout);
     }
 
     /* Phase B: tell others to exit themselves */
     if (me == ROOT_PE) {
         shmem_int_p(ROOT_GO, 1, ROOT_PE);  /* publish GO=1 on root */
         shmem_quiet();                      /* ensure visibility */
 
         if (g_debug) {
             double elapsed_ms = (now_sec() - g_start_time) * 1e3;
             printf("PE %d (root) releasing non-roots; will exit last (t=%.3f ms)\n", me, elapsed_ms);
             fflush(stdout);
         }
 
         /* small grace period to let others stop polling us */
         struct timespec ts = {0, 2 * 1000 * 1000}; /* 2 ms */
         nanosleep(&ts, NULL);
 
         shmem_quiet();
         shmem_global_exit(0);
     }
 }
 
 /* ---------- core algorithm ---------- */
 
 /* Try to mark the leaf group as done (first-come-first-serve among any PE in the leaf group).
  * Returns 1 if this call set the flag (won CAS) or the flag was already set; 0 otherwise. */
 static int try_mark_leaf_group_done(int me, int npes) {
     int level = 0;
     int span  = group_span_at_level(G_LEAF, level);
     int gidx  = me / span;
     int g_leader = group_leader_pe(G_LEAF, level, gidx);
 
     /* Determine this leaf group's PE range */
     int start = g_leader;
     int end   = start + span;
     if (end > npes) end = npes;
 
     /* Check if ALL LOCAL_DONE in the group == -1 */
     int all_done = 1;
     for (int pe = start; pe < end; pe++) {
         int v = (pe == me) ? *LOCAL_DONE : shmem_int_g(LOCAL_DONE, pe);
         if (v != -1) { all_done = 0; break; }
     }
 
     if (!all_done) return 0;
 
     /* Atomically set leaf group flag at the group's leader PE */
     int old = shmem_int_atomic_compare_swap(&GROUP_DONE[level][gidx], 0, 1, g_leader);
     if (g_debug && (old == 0 || shmem_int_g(&GROUP_DONE[level][gidx], g_leader) == 1)) {
         printf("PE %d observed LEAF group %d done; flag set at leader PE %d\n", me, gidx, g_leader);
         fflush(stdout);
     }
     /* Treat already-set as success too */
     return 1;
 }
 
 /* Attempt to propagate done flags up the binary tree.
  * Any PE may help, but only the designated group leaders will "own" and write parent flags.
  * Root leader (PE 0) will coordinate a two-phase exit once top-level flag is set. */
 static void propagate_up_and_maybe_exit(void) {
     int me   = shmem_my_pe();
     int npes = shmem_n_pes();
 
     /* Keep attempting until global exit happens (root will call it last) */
     while (1) {
         /* EARLY CHECK: if top flag is set, stop all work immediately */
         int top_level = MAX_LEVELS - 1;
         int top_flag  = shmem_int_g(&GROUP_DONE[top_level][0], ROOT_PE);
 
         if (top_flag == 1) {
             if (me == ROOT_PE) {
                 /* root prints, then releases others, then exits last */
                 root_print_then_release_and_exit();
                 /* not reached */
             } else {
                 /* Wait for root to flip GO, then exit ourselves. Avoid other RMAs. */
                 while (shmem_int_g(ROOT_GO, ROOT_PE) == 0) {
                     tiny_pause();
                 }
                 shmem_quiet();
                 shmem_global_exit(0);
                 /* not reached */
             }
         }
 
         /* 1) First, try to set our leaf group if possible */
         (void) try_mark_leaf_group_done(me, npes);
 
         /* 2) For each internal level, if I'm that level's group leader,
               and both child groups are done, set my parent flag. */
         for (int L = 1; L < MAX_LEVELS; L++) {
             int span_here = group_span_at_level(G_LEAF, L);
             if (me % span_here != 0) continue; /* I'm not leader at this level */
 
             int my_gidx   = me / span_here;
             int childL    = left_child_idx(my_gidx);
             int childR    = right_child_idx(my_gidx);
 
             if (childL >= NUM_GROUPS[L-1]) continue; /* no children at left -> nothing to do */
 
             int left_leader  = group_leader_pe(G_LEAF, L-1, childL);
             int left_done    = shmem_int_g(&GROUP_DONE[L-1][childL], left_leader);
 
             int right_done = 1;
             int right_leader = -1;
             if (childR < NUM_GROUPS[L-1]) {
                 right_leader = group_leader_pe(G_LEAF, L-1, childR);
                 right_done   = shmem_int_g(&GROUP_DONE[L-1][childR], right_leader);
             }
 
             if (left_done && right_done) {
                 /* Mark my current level group as done (i.e., my group's flag) */
                 int old_me_flag = shmem_int_atomic_compare_swap(&GROUP_DONE[L][my_gidx], 0, 1, me);
                 if (g_debug && old_me_flag == 0) {
                     printf("PE %d (leader L=%d,g=%d) set its OWN group-done flag\n",
                            me, L, my_gidx);
                     fflush(stdout);
                 }
 
                 /* If not root level yet, try to mark parent (at level L+1) */
                 if (L + 1 < MAX_LEVELS) {
                     int parent_idx    = my_gidx / 2;
                     int parent_leader = group_leader_pe(G_LEAF, L+1, parent_idx);
                     (void) shmem_int_atomic_compare_swap(&GROUP_DONE[L+1][parent_idx], 0, 1, parent_leader);
                     if (g_debug && me == parent_leader) {
                         printf("PE %d helped propagate to parent (L=%d,g=%d)\n",
                                me, L+1, parent_idx);
                         fflush(stdout);
                     }
                 }
             }
         }
 
         tiny_pause();
     }
 }
 
 /* ---------- main ---------- */
 
 int main(int argc, char **argv) {
     shmem_init();
     int me   = shmem_my_pe();
     int npes = shmem_n_pes();
 
     g_debug = env_debug_enabled();
     G_LEAF  = env_group_size();
 
     /* Align start for timing; not required for logic */
     shmem_barrier_all();
     g_start_time = now_sec();
 
     /* Symmetric allocations */
     LOCAL_DONE  = shmem_malloc(sizeof(int));
     ELAPSED_MS  = shmem_malloc(sizeof(double));
     AGG_PRINTED = shmem_malloc(sizeof(int));
     ROOT_GO     = shmem_malloc(sizeof(int));
     if (!LOCAL_DONE || !ELAPSED_MS || !AGG_PRINTED || !ROOT_GO) shmem_global_exit(1);
 
     *LOCAL_DONE  = 0;
     *ELAPSED_MS  = 0.0;
     *AGG_PRINTED = 0;
     *ROOT_GO     = 0;
 
     allocate_tree_flags(npes);
 
     /* Mark local done and timestamp */
     *LOCAL_DONE = -1;
     *ELAPSED_MS = (now_sec() - g_start_time) * 1e3;
 
     if (g_debug && me == 0) {
         printf("[DEBUG] npes=%d, leaf_group_size=%d, levels=%d\n",
                npes, G_LEAF, MAX_LEVELS);
         for (int L = 0; L < MAX_LEVELS; L++) {
             printf("[DEBUG]  level %d: num_groups=%d, span=%d, leaders: ",
                    L, NUM_GROUPS[L], group_span_at_level(G_LEAF, L));
             for (int g = 0; g < NUM_GROUPS[L]; g++) {
                 printf("%d ", group_leader_pe(G_LEAF, L, g));
             }
             printf("\n");
             fflush(stdout);
         }
     }
 
     /* Everyone participates in propagation; root coordinates exit last */
     propagate_up_and_maybe_exit();
 
     /* Not reached (root exits the job for all PEs). */
     shmem_finalize();
     return 0;
 }
 