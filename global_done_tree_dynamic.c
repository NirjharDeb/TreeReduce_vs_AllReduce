/* global_done_tree_dynamic.c
 *
 * Tree-based global termination for OpenSHMEM with DYNAMIC leaders.
 * - The "leader" of each group (at every level) is chosen dynamically as the PE
 *   that *successfully completes that group's done CAS*:
 *     - At LEAF level, this is the PE that increments the group's completion
 *       counter to the group size (i.e., the last finisher in the leaf).
 *     - At INTERNAL levels, this is the PE that first observes both children
 *       done and successfully CASes the group's flag from 0->1.
 * - Group flags are still *hosted* at the canonical static group owner
 *   (first PE of the group's span) to keep addressing simple, but the PE
 *   that *acts as leader* is dynamic and stored in GROUP_LEADER[L][g].
 * - Root (PE 0) still coordinates two-phase exit/printing.
 *
 * Compile:
 *   oshcc -O3 -std=c11 -o global_done_tree_dynamic global_done_tree_dynamic.c
 *
 * Run:
 *   srun --mpi=pmix -N 1 -n 24 ./global_done_tree_dynamic
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
 static long   *EXIT_ACKS;       /* on ROOT_PE: number of non-roots that acknowledged and will exit */
 
 static int   **GROUP_DONE;      /* per level array of group flags (hosted at static group owner) */
 static int   **GROUP_LEADER;    /* per level, dynamic leader PE id (hosted at static group owner) */
 static int     MAX_LEVELS;      /* number of levels including root level */
 static int    *NUM_GROUPS;      /* number of groups at each level */
 
 /* Leaf-level completion counters (hosted at static leaf owner PE) */
 static int    *LEAF_COUNT;
 
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
 
 /* canonical (static) group owner PE id for (level, group_idx) */
 static inline int static_group_owner_pe(int leaf_size, int level, int group_idx) {
     return group_idx * group_span_at_level(leaf_size, level);
 }
 
 /* child group indices given parent index */
 static inline int left_child_idx(int parent_idx)  { return parent_idx * 2; }
 static inline int right_child_idx(int parent_idx) { return parent_idx * 2 + 1; }
 
 /* ---------- allocation of symmetric arrays per level ---------- */
 static void allocate_tree_flags(int npes) {
     /* Determine number of levels until root (1 group) */
     int levels = 0;
     while (1) {
         int ng = num_groups_at_level(npes, G_LEAF, levels);
         levels++;
         if (ng <= 1) break;
     }
     MAX_LEVELS = levels; /* includes the top (root) level */
 
     NUM_GROUPS   = shmem_malloc(sizeof(int)   * MAX_LEVELS);
     GROUP_DONE   = shmem_malloc(sizeof(int*)  * MAX_LEVELS);
     GROUP_LEADER = shmem_malloc(sizeof(int*)  * MAX_LEVELS);
     if (!NUM_GROUPS || !GROUP_DONE || !GROUP_LEADER) {
         shmem_global_exit(1);
     }
 
     for (int L = 0; L < MAX_LEVELS; L++) {
         int ng = num_groups_at_level(npes, G_LEAF, L);
         NUM_GROUPS[L]   = ng;
         GROUP_DONE[L]   = shmem_malloc(sizeof(int) * ng);
         GROUP_LEADER[L] = shmem_malloc(sizeof(int) * ng);
         if (!GROUP_DONE[L] || !GROUP_LEADER[L]) shmem_global_exit(1);
         for (int i = 0; i < ng; i++) {
             GROUP_DONE[L][i]   = 0;   /* 0 = not done, 1 = done */
             GROUP_LEADER[L][i] = -1;  /* unknown until decided dynamically */
         }
     }
 
     /* Leaf counters */
     LEAF_COUNT = shmem_malloc(sizeof(int) * NUM_GROUPS[0]);
     if (!LEAF_COUNT) shmem_global_exit(1);
     for (int i = 0; i < NUM_GROUPS[0]; i++) LEAF_COUNT[i] = 0;
 }
 
 /* Spin helper: small backoff to avoid hammering */
 static inline void tiny_pause(void) {
     struct timespec ts = {0, 1000000}; /* 1 ms */
     nanosleep(&ts, NULL);
 }
 
 /* ---------- root print + coordinated exit (with ACKs) ---------- */
 
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
 
     /* Release non-roots to exit, then wait for ACKs from all of them. */
     if (me == ROOT_PE) {
         /* Publish GO = 1 */
         shmem_int_p(ROOT_GO, 1, ROOT_PE);
         shmem_quiet();
 
         if (g_debug) {
             double elapsed_ms = (now_sec() - g_start_time) * 1e3;
             printf("PE %d (root) released non-roots; waiting for %d ACKs (t=%.3f ms)\n",
                    me, npes - 1, elapsed_ms);
             fflush(stdout);
         }
 
         /* Wait for all non-roots to acknowledge they are exiting */
         while (shmem_long_g(EXIT_ACKS, ROOT_PE) < (long)(npes - 1)) {
             tiny_pause();
         }
 
         if (g_debug) {
             double elapsed_ms = (now_sec() - g_start_time) * 1e3;
             printf("PE %d (root) received all ACKs; exiting last (t=%.3f ms)\n", me, elapsed_ms);
             fflush(stdout);
         }
 
         shmem_quiet();
         shmem_global_exit(0);
     }
 }
 
 /* ---------- core algorithm ---------- */
 
 /* Try to mark the leaf group as done using a *dynamic* leader:
  * Each PE increments the group's leaf counter at the group's static owner PE.
  * The PE that observes its fetch_inc return == (group_size - 1) is the last finisher,
  * becomes the dynamic leader for this leaf group, and sets the leaf GROUP_DONE flag. */
 static int try_mark_leaf_group_done(int me, int npes) {
     const int level = 0;
     const int span  = group_span_at_level(G_LEAF, level);
     const int gidx  = me / span;
     const int host  = static_group_owner_pe(G_LEAF, level, gidx);
 
     /* Determine this leaf group's PE range to compute actual size (tail groups). */
     int start = host;
     int end   = start + span;
     if (end > npes) end = npes;
     const int gsize = end - start;
 
     /* If already marked done, nothing to do. */
     if (shmem_int_g(&GROUP_DONE[level][gidx], host) == 1) return 1;
 
     /* Increment the group's completion counter at the static host. */
     int prior = shmem_int_atomic_fetch_inc(&LEAF_COUNT[gidx], host);
 
     if (prior == gsize - 1) {
         /* I'm the last finisher in this leaf group -> dynamic leader */
         (void) shmem_int_atomic_compare_swap(&GROUP_DONE[level][gidx], 0, 1, host);
         shmem_int_p(&GROUP_LEADER[level][gidx], me, host);
         if (g_debug) {
             printf("PE %d is DYNAMIC LEAF LEADER for group %d (size=%d); flag set at host %d\n",
                    me, gidx, gsize, host);
             fflush(stdout);
         }
         return 1;
     }
     return 0;
 }
 
 /* Attempt to propagate done flags up the binary tree.
  * Allow seeding of a leader when none exists; otherwise only the dynamic leader acts.
  * Group flags/leaders are hosted at the group's static owner PE to keep addressing stable. */
 static void propagate_up_and_maybe_exit(void) {
     int me   = shmem_my_pe();
     int npes = shmem_n_pes();
 
     while (1) {
         /* EARLY CHECK: if top flag is set, stop all work immediately */
         int top_level = MAX_LEVELS - 1;
         int top_flag  = shmem_int_g(&GROUP_DONE[top_level][0], ROOT_PE);
 
         if (top_flag == 1) {
             if (me == ROOT_PE) {
                 root_print_then_release_and_exit();
                 /* not reached */
             } else {
                 while (shmem_int_g(ROOT_GO, ROOT_PE) == 0) {
                     tiny_pause();
                 }
                 (void) shmem_long_atomic_fetch_inc(EXIT_ACKS, ROOT_PE);
                 shmem_quiet();
                 shmem_global_exit(0);
                 /* not reached */
             }
         }
 
         /* 1) Try to complete our leaf group (this will elect the dynamic leader). */
         (void) try_mark_leaf_group_done(me, npes);
 
         /* 2) Internal levels: allow seeding when no leader yet; otherwise only the
          *    dynamic leader acts. */
         for (int L = 1; L < MAX_LEVELS; L++) {
             int span_here   = group_span_at_level(G_LEAF, L);
             int my_gidx     = me / span_here;
             int static_host = static_group_owner_pe(G_LEAF, L, my_gidx);
 
             /* If a leader exists and it's not me, skip. If no leader yet (-1), I may try to seed. */
             int cur_leader = shmem_int_g(&GROUP_LEADER[L][my_gidx], static_host);
             if (cur_leader != -1 && cur_leader != me) continue;
 
             /* Check children done (read from their static hosts). */
             int childL = left_child_idx(my_gidx);
             int childR = right_child_idx(my_gidx);
             if (childL >= NUM_GROUPS[L-1]) continue;
 
             int left_host = static_group_owner_pe(G_LEAF, L-1, childL);
             int left_done = shmem_int_g(&GROUP_DONE[L-1][childL], left_host);
 
             int right_done = 1;
             if (childR < NUM_GROUPS[L-1]) {
                 int right_host = static_group_owner_pe(G_LEAF, L-1, childR);
                 right_done = shmem_int_g(&GROUP_DONE[L-1][childR], right_host);
             }
 
             if (left_done && right_done) {
                 /* Try to complete this group; whoever wins CAS becomes dynamic leader. */
                 int old = shmem_int_atomic_compare_swap(&GROUP_DONE[L][my_gidx], 0, 1, static_host);
                 if (old == 0) {
                     /* I just completed this group: seed/overwrite the leader with me. */
                     shmem_int_p(&GROUP_LEADER[L][my_gidx], me, static_host);
                     if (g_debug) {
                         printf("PE %d became DYNAMIC leader at L=%d,g=%d (host=%d)\n",
                                me, L, my_gidx, static_host);
                         fflush(stdout);
                     }
 
                     /* If not at the root, opportunistically try to mark parent and seed its leader. */
                     if (L + 1 < MAX_LEVELS) {
                         int parent_idx  = my_gidx / 2;
                         int parent_host = static_group_owner_pe(G_LEAF, L+1, parent_idx);
                         int pold = shmem_int_atomic_compare_swap(&GROUP_DONE[L+1][parent_idx], 0, 1, parent_host);
                         if (pold == 0) {
                             shmem_int_p(&GROUP_LEADER[L+1][parent_idx], me, parent_host);
                             if (g_debug) {
                                 printf("PE %d set PARENT L=%d,g=%d done and became its DYNAMIC leader (host=%d)\n",
                                        me, L+1, parent_idx, parent_host);
                                 fflush(stdout);
                             }
                         }
                     }
                 } else {
                     /* Someone else completed it. If leader still unset, they'll set it soon. */
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
     EXIT_ACKS   = shmem_malloc(sizeof(long));
     if (!LOCAL_DONE || !ELAPSED_MS || !AGG_PRINTED || !ROOT_GO || !EXIT_ACKS) shmem_global_exit(1);
 
     *LOCAL_DONE  = 0;
     *ELAPSED_MS  = 0.0;
     *AGG_PRINTED = 0;
     *ROOT_GO     = 0;
     *EXIT_ACKS   = 0;
 
     allocate_tree_flags(npes);
 
     /* Mark local done and timestamp */
     *LOCAL_DONE = -1;
     *ELAPSED_MS = (now_sec() - g_start_time) * 1e3;
 
     if (g_debug && me == 0) {
         printf("[DEBUG] npes=%d, leaf_group_size=%d, levels=%d\n",
                npes, G_LEAF, MAX_LEVELS);
         for (int L = 0; L < MAX_LEVELS; L++) {
             printf("[DEBUG]  level %d: num_groups=%d, span=%d\n",
                    L, NUM_GROUPS[L], group_span_at_level(G_LEAF, L));
         }
         fflush(stdout);
     }
 
     /* Everyone participates in propagation; root coordinates exit last */
     propagate_up_and_maybe_exit();
 
     /* Not reached (root exits the job for all PEs). */
     shmem_finalize();
     return 0;
 }
 