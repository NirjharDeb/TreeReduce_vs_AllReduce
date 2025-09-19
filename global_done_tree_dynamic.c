/* global_done_tree_dynamic.c
 *
 * Tree-based global termination for OpenSHMEM with DYNAMIC leaders.
 * - The "leader" of each group (at every level) is the last to finish within that group:
 *     - At LEAF level, detected by fetch_inc reaching (group_size-1).
 *     - At INTERNAL levels, detected by the second (or only) child to finish.
 * - Group flags are *hosted* at the canonical static group owner for addressing,
 *   but the acting leader is dynamic and stored in GROUP_LEADER[L][g].
 * - Root (PE 0) coordinates the final two-phase exit/printing.
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
 
 /* NEW: per-parent child completion counters (levels 1..MAX_LEVELS-1) */
 static int   **CHILD_DONE_COUNT;
 
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
 
     /* NEW: child completion counters for parents at levels 1..MAX_LEVELS-1 */
     CHILD_DONE_COUNT = shmem_malloc(sizeof(int*) * MAX_LEVELS);
     if (!CHILD_DONE_COUNT) shmem_global_exit(1);
     CHILD_DONE_COUNT[0] = NULL; /* unused for leaves */
     for (int L = 1; L < MAX_LEVELS; L++) {
         int ng = NUM_GROUPS[L];
         CHILD_DONE_COUNT[L] = shmem_malloc(sizeof(int) * ng);
         if (!CHILD_DONE_COUNT[L]) shmem_global_exit(1);
         for (int i = 0; i < ng; i++) CHILD_DONE_COUNT[L][i] = 0;
     }
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
 
         /* Efficient local wait on the root's EXIT_ACKS */
         shmem_long_wait_until(EXIT_ACKS, SHMEM_CMP_GE, (long)(npes - 1));
 
         if (g_debug) {
             double elapsed_ms = (now_sec() - g_start_time) * 1e3;
             printf("PE %d (root) received all ACKs; exiting last (t=%.3f ms)\n", me, elapsed_ms);
             fflush(stdout);
         }
 
         shmem_quiet();
         shmem_global_exit(0);
     }
 }
 
 /* -------- NEW: complete a group and (if last child) propagate upward -------- */
 static void complete_group_and_maybe_propagate(int me, int L, int gidx, int npes) {
     /* Set this group's done flag and leader (hosted at its static owner) */
     int host = static_group_owner_pe(G_LEAF, L, gidx);
     (void) shmem_int_atomic_compare_swap(&GROUP_DONE[L][gidx], 0, 1, host);
     shmem_int_p(&GROUP_LEADER[L][gidx], me, host);
 
     if (g_debug) {
         printf("PE %d finalized L=%d,g=%d (host=%d) as dynamic leader\n", me, L, gidx, host);
         fflush(stdout);
     }
 
     /* Walk up while we are the LAST finishing child at each parent */
     while (L + 1 < MAX_LEVELS) {
         int parent_L   = L + 1;
         int parent_idx = gidx / 2;
         int parent_host = static_group_owner_pe(G_LEAF, parent_L, parent_idx);
 
         /* Determine how many children this parent has (1 for tail, else 2) */
         int childL = left_child_idx(parent_idx);
         int childR = right_child_idx(parent_idx);
         int expected_children = 1 + (childR < NUM_GROUPS[L] ? 1 : 0);
 
         /* Atomically record that this child finished */
         int prior = shmem_int_atomic_fetch_inc(&CHILD_DONE_COUNT[parent_L][parent_idx], parent_host);
 
         if (prior + 1 == expected_children) {
             /* We are the LAST child to finish => we become parent's dynamic leader */
             (void) shmem_int_atomic_compare_swap(&GROUP_DONE[parent_L][parent_idx], 0, 1, parent_host);
             shmem_int_p(&GROUP_LEADER[parent_L][parent_idx], me, parent_host);
 
             if (g_debug) {
                 printf("PE %d became DYNAMIC leader at L=%d,g=%d (last child; host=%d)\n",
                        me, parent_L, parent_idx, parent_host);
                 fflush(stdout);
             }
 
             /* Move up a level and repeat */
             L    = parent_L;
             gidx = parent_idx;
             host = parent_host;
             continue;
         } else {
             /* Not the last child at this parent; stop propagating here */
             break;
         }
     }
 }
 
 /* ---------- core algorithm ---------- */
 
 /* Try to mark the leaf group as done using a *dynamic* leader:
  * Each PE increments the group's leaf counter at the group's static owner PE.
  * The PE that observes its fetch_inc return == (group_size - 1) is the last finisher,
  * becomes the dynamic leader for this leaf group, completes the group, and
  * (if last child at parent) propagates upward via CHILD_DONE_COUNT. */
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
         /* I'm the last finisher in this leaf group -> I complete it and maybe propagate */
         complete_group_and_maybe_propagate(me, /*L=*/0, gidx, npes);
         return 1;
     }
     return 0;
 }
 
 /* Attempt to propagate done flags up the binary tree.
  * INTERNAL groups are now completed exclusively by the LAST child to finish
  * (via complete_group_and_maybe_propagate), so we no longer do "anyone can CAS"
  * based on reading both children done. */
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
                 /* Poll the ROOT_PE's ROOT_GO remotely; wait_until on local would hang */
                 while (shmem_int_g(ROOT_GO, ROOT_PE) == 0) {
                     tiny_pause();
                 }
                 (void) shmem_long_atomic_fetch_inc(EXIT_ACKS, ROOT_PE);
                 shmem_quiet();
                 shmem_global_exit(0);
                 /* not reached */
             }
         }
 
         /* Leaves elect leaders and trigger upward propagation when last in their group */
         (void) try_mark_leaf_group_done(me, npes);
 
         /* No internal "helping" CAS here—internal completion is driven by last-child promotion. */
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
 