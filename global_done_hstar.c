/* global_done_hstar.c
 *
 * H-STAR (multi-level STAR) global termination for OpenSHMEM (no atomics).
 *
 * Model (0 = not done, -1 = done):
 * 1) Each PE writes -1 to its *group-local slot* at the level-0 group's anchor.
 * 2) Each level-0 anchor waits for its members, then marks its parent group's slot at level 1.
 * 3) This repeats up the levels until the top-level root proves global completion.
 * 4) The root initiates a tree broadcast of a single global flag (-1) downward.
 * 5) Everyone waits on the per-PE global flag, hits a final barrier, and exits cleanly.
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
 static int    *LOCAL_DONE;                 /* per-PE local flag: -1 = done, 0 = not done */
 static double *ELAPSED_MS;                 /* per-PE elapsed time (ms) */
 
 /* STAR scheme configuration/state */
 static int     G_LEAF = 8;                 /* group size (can be changed via env) */
 static int     NUM_GROUPS0 = 0;            /* number of groups at "leaf" granularity */
 static int     g_debug = 0;
 static double  g_start_time = 0.0;
 static const int ROOT_PE = 0;
 
 /* Per-group, per-member completion flags at the group's anchor:
  * GROUP_PE_DONE[g][member] == -1 means that member PE has finished. */
 static int   **GROUP_PE_DONE;              /* shape: [NUM_GROUPS0][G_LEAF] */
 
 /* Root’s record that each group has finished: ROOT_GROUP_DONE[g] == -1 means group g is done.
  * (Retained for backward compatibility; not used by H-STAR.) */
 static int    *ROOT_GROUP_DONE;            /* length: NUM_GROUPS0, authoritative at ROOT_PE */
 
 /* Global termination gate: set/broadcast to -1 by the root (hierarchically in H-STAR). */
 static int    *GLOBAL_TERMINATION_READY;   /* each PE waits on its local copy */
 
 /* ---------- H-STAR additions (minimal) ---------- */
 
 /* H-STAR: branching factor above the leaf (children per owner at levels >= 1) */
 static int K = 8;
 
 /* H-STAR: total number of levels (level 0 = leaf; level LEVELS-1 = root) */
 static int LEVELS = 1;
 
 /* H-STAR: per-level group counts; NUM_GROUPS[0] == NUM_GROUPS0 */
 static int *NUM_GROUPS;                    /* length LEVELS */
 
 /* H-STAR: child completion mailboxes per level.
  * For l=0 this *aliases* GROUP_PE_DONE to stay close to original layout/style. */
 static int ***LVL_CHILD_DONE;              /* [LEVELS][NUM_GROUPS[l]][child_cap(l)] */
 
 /* H-STAR: per-(level,group) downward broadcast token.
  * Owners wait on their local token, then forward to children (owners or PEs). */
 static int  **LVL_BCAST_TOKEN;             /* [LEVELS][NUM_GROUPS[l]] */
 
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
 
 /* H-STAR: read branch factor K from env */
 static int env_branch_k(void) {
     const char *e = getenv("GLOBAL_BRANCH_K");
     if (!e || e[0] == '\0') return 8;
     int v = atoi(e);
     return (v >= 2) ? v : 8;
 }
 
 static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
 
 /* H-STAR: integer power for spans (K^level) */
 static inline int ipow(int base, int exp) {
     int r = 1;
     while (exp-- > 0) r *= base;
     return r;
 }
 
 /* size (in PEs) of a group at level L (L=0 is leaf) */
 /* H-STAR: was leaf_size * 2^level; generalize to leaf_size * K^level */
 static inline int group_span_at_level(int leaf_size, int level) {
     return leaf_size * ipow(K, level);
 }
 
 /* canonical (static) group owner PE id for (level, group_idx) */
 /* H-STAR: use the proper level span (kept signature/style) */
 static inline int static_group_owner_pe(int leaf_size, int level, int group_idx) {
     return group_idx * group_span_at_level(leaf_size, level);
 }
 
 /* ---------- H-STAR planning/allocation ---------- */
 
 /* H-STAR: compute number of levels and groups per level */
 static void compute_levels_and_groups(int npes) {
     int ng0 = ceil_div(npes, G_LEAF);
     /* Count levels until exactly one group remains at the top. */
     int levels = 1;
     int prev = ng0;
     while (prev > 1) { prev = ceil_div(prev, K); levels++; }
 
     LEVELS = levels;
 
     NUM_GROUPS = shmem_malloc(sizeof(int) * LEVELS);
     if (!NUM_GROUPS) shmem_global_exit(1);
 
     NUM_GROUPS[0] = ng0;
     for (int l = 1; l < LEVELS; l++) NUM_GROUPS[l] = ceil_div(NUM_GROUPS[l-1], K);
 
     NUM_GROUPS0 = NUM_GROUPS[0]; /* maintain original debug/printing variable */
 }
 
 /* ---------- STAR/H-STAR allocation ---------- */
 static void allocate_star_flags(int npes) {
     /* H-STAR: compute hierarchy first (kept function name to minimize churn) */
     compute_levels_and_groups(npes);
 
     /* Level-0 per-group per-member flags at group anchors (as before). */
     GROUP_PE_DONE = shmem_malloc(sizeof(int*) * NUM_GROUPS0);
     if (!GROUP_PE_DONE) shmem_global_exit(1);
     for (int g = 0; g < NUM_GROUPS0; g++) {
         GROUP_PE_DONE[g] = shmem_malloc(sizeof(int) * G_LEAF);
         if (!GROUP_PE_DONE[g]) shmem_global_exit(1);
         for (int i = 0; i < G_LEAF; i++) GROUP_PE_DONE[g][i] = 0; /* 0 = not done */
     }
 
     /* Root’s per-group record (retained for compatibility; H-STAR doesn't use it). */
     ROOT_GROUP_DONE = shmem_malloc(sizeof(int) * NUM_GROUPS0);
     if (!ROOT_GROUP_DONE) shmem_global_exit(1);
     for (int g = 0; g < NUM_GROUPS0; g++) ROOT_GROUP_DONE[g] = 0;
 
     /* Global termination flag */
     GLOBAL_TERMINATION_READY = shmem_malloc(sizeof(int));
     if (!GLOBAL_TERMINATION_READY) shmem_global_exit(1);
     *GLOBAL_TERMINATION_READY = 0;
 
     /* H-STAR: allocate per-level child mailboxes and per-level broadcast tokens. */
     LVL_CHILD_DONE = shmem_malloc(sizeof(int**) * LEVELS);
     LVL_BCAST_TOKEN = shmem_malloc(sizeof(int*) * LEVELS);
     if (!LVL_CHILD_DONE || !LVL_BCAST_TOKEN) shmem_global_exit(1);
 
     for (int l = 0; l < LEVELS; l++) {
         const int groups = NUM_GROUPS[l];
         const int cap    = (l == 0) ? G_LEAF : K;
 
         /* child mailboxes: [groups][cap], initialized to 0 */
         LVL_CHILD_DONE[l] = shmem_malloc(sizeof(int*) * groups);
         if (!LVL_CHILD_DONE[l]) shmem_global_exit(1);
         for (int g = 0; g < groups; g++) {
             LVL_CHILD_DONE[l][g] = shmem_malloc(sizeof(int) * cap);
             if (!LVL_CHILD_DONE[l][g]) shmem_global_exit(1);
             for (int i = 0; i < cap; i++) LVL_CHILD_DONE[l][g][i] = 0;
         }
 
         /* downward token: one int per group, initialized to 0 */
         LVL_BCAST_TOKEN[l] = shmem_malloc(sizeof(int) * groups);
         if (!LVL_BCAST_TOKEN[l]) shmem_global_exit(1);
         for (int g = 0; g < groups; g++) LVL_BCAST_TOKEN[l][g] = 0;
     }
 
     /* H-STAR: alias level-0 child-done to the original name for readability. */
     /* (No change in usage style at the leaf.) */
     for (int g = 0; g < NUM_GROUPS0; g++) {
         /* GROUP_PE_DONE[g] already allocated above; ensure LVL view points to it. */
         /* We reassign the level-0 pointer to reuse the same buffers. */
         /* (This keeps leaf-level code intuitive.) */
         LVL_CHILD_DONE[0][g] = GROUP_PE_DONE[g];
     }
 }
 
 /* ---------- H-STAR termination protocol ---------- */
 static void run_hstar_termination(void) {
     const int me   = shmem_my_pe();
     const int npes = shmem_n_pes();
 
     /* ----- local completion (unchanged) ----- */
     const int g0   = me / G_LEAF;          /* my leaf group */
     const int idx0 = me % G_LEAF;          /* index within leaf group */
     const int own0 = static_group_owner_pe(G_LEAF, /*level=*/0, g0);
 
     *LOCAL_DONE = -1;
     *ELAPSED_MS = (now_sec() - g_start_time) * 1e3;
 
     /* Leaf: PUT -1 into my slot at my level-0 group owner. */
     shmem_int_p(&LVL_CHILD_DONE[0][g0][idx0], -1, own0);
     shmem_quiet();
 
     /* ----- upward fan-in across levels ----- */
     for (int l = 0; l < LEVELS; l++) {
         /* If I am the owner of my level-l group, wait on its children,
          * then (if not at the top) notify my parent. */
         const int span_l  = group_span_at_level(G_LEAF, l);
         const int g_l     = me / span_l;
         const int owner_l = static_group_owner_pe(G_LEAF, l, g_l);
 
         if (me == owner_l) {
             /* Determine actual child count for this group at level l. */
             int gsize;
             if (l == 0) {
                 int start = owner_l;
                 int end   = start + G_LEAF;
                 if (end > npes) end = npes;
                 gsize = end - start;
             } else {
                 /* children are level-(l-1) owners */
                 const int groups_below = NUM_GROUPS[l-1];
                 const int first_child  = g_l * K;
                 const int max_child    = first_child + K;
                 gsize = max_child <= groups_below ? K : (groups_below - first_child);
                 if (gsize < 0) gsize = 0;
             }
 
             /* Wait for all children at this level. */
             for (int i = 0; i < gsize; i++) {
                 shmem_int_wait_until(&LVL_CHILD_DONE[l][g_l][i], SHMEM_CMP_EQ, -1);
             }
 
             /* If not top, notify my parent owner at level (l+1). */
             if (l + 1 < LEVELS) {
                 const int parent_l     = l + 1;
                 const int parent_g     = g_l / K;
                 const int parent_owner = static_group_owner_pe(G_LEAF, parent_l, parent_g);
                 const int my_child_idx = g_l % K; /* my slot among parent's children */
                 shmem_int_p(&LVL_CHILD_DONE[parent_l][parent_g][my_child_idx], -1, parent_owner);
                 shmem_quiet();
             }
         }
     }
 
     /* ----- root proves completion and starts downward broadcast ----- */
     if (me == ROOT_PE) {
         /* At the top level, there is exactly 1 group: g = 0. */
         const int top_l = LEVELS - 1;
         const int top_g = 0;
 
         /* Optional: aggregate timing like STAR (same style). */
         double sum = 0.0, minv = 0.0, maxv = 0.0;
         for (int pe = 0; pe < npes; pe++) {
             double val = (pe == me) ? *ELAPSED_MS : shmem_double_g(ELAPSED_MS, pe);
             if (pe == 0) { minv = maxv = val; }
             if (val < minv) minv = val;
             if (val > maxv) maxv = val;
             sum += val;
         }
         double avg = sum / (double)npes;
         printf("ELAPSED_MS across %d PEs: min=%.3f ms  avg=%.3f ms  max=%.3f ms\n",
                npes, minv, avg, maxv);
         fflush(stdout);
 
         /* H-STAR: seed the tree broadcast by setting the top group's token. */
         LVL_BCAST_TOKEN[top_l][top_g] = -1; /* local store at root owner */
     }
 
     /* ----- downward fan-out as a tree (owners forward tokens) ----- */
     /* Owners at each level wait on their token, then forward to children owners (or PEs at leaf). */
     for (int l = LEVELS - 1; l >= 0; l--) {
         const int span_l  = group_span_at_level(G_LEAF, l);
         const int g_l     = me / span_l;
         const int owner_l = static_group_owner_pe(G_LEAF, l, g_l);
 
         if (me == owner_l) {
             /* Wait until my group's token is set by my parent (or root). */
             shmem_int_wait_until(&LVL_BCAST_TOKEN[l][g_l], SHMEM_CMP_EQ, -1);
 
             if (l > 0) {
                 /* Forward to child owners at level (l-1). */
                 const int groups_below = NUM_GROUPS[l-1];
                 const int first_child  = g_l * K;
                 int gsize = (first_child + K <= groups_below) ? K : (groups_below - first_child);
                 if (gsize < 0) gsize = 0;
 
                 for (int c = 0; c < gsize; c++) {
                     const int child_g     = first_child + c;
                     const int child_owner = static_group_owner_pe(G_LEAF, l - 1, child_g);
                     shmem_int_p(&LVL_BCAST_TOKEN[l - 1][child_g], -1, child_owner);
                 }
                 shmem_quiet();
             } else {
                 /* Leaf owners: set each member PE's per-PE gate. */
                 int start = owner_l;
                 int end   = start + G_LEAF;
                 if (end > npes) end = npes;
                 for (int pe = start; pe < end; pe++) {
                     if (pe == owner_l) {
                         *GLOBAL_TERMINATION_READY = -1; /* local store for owner */
                     } else {
                         shmem_int_p(GLOBAL_TERMINATION_READY, -1, pe);
                     }
                 }
                 shmem_quiet();
             }
         }
     }
 
     /* ----- all PEs wait for the global gate, then finalize (unchanged) ----- */
     shmem_int_wait_until(GLOBAL_TERMINATION_READY, SHMEM_CMP_EQ, -1);
     shmem_barrier_all();
 
     if (me == ROOT_PE) {
         const int np = shmem_n_pes();
         printf("ALL_CLEAR: all %d PEs observed termination and reached the final barrier.\n", np);
         fflush(stdout);
     }
 }
 
 /* ---------- main ---------- */
 
 int main(int argc, char **argv) {
     shmem_init();
     int me   = shmem_my_pe();
     int npes = shmem_n_pes();
 
     g_debug = env_debug_enabled();
     G_LEAF  = env_group_size();
     K       = env_branch_k();            /* H-STAR: branch factor from env (GLOBAL_BRANCH_K) */
 
     /* Align start for timing; not required for logic */
     shmem_barrier_all();
     g_start_time = now_sec();
 
     /* Symmetric allocations (local bookkeeping + H-STAR flags) */
     LOCAL_DONE = shmem_malloc(sizeof(int));
     ELAPSED_MS = shmem_malloc(sizeof(double));
     if (!LOCAL_DONE || !ELAPSED_MS) shmem_global_exit(1);
 
     *LOCAL_DONE = 0;
     *ELAPSED_MS = 0.0;
 
     allocate_star_flags(npes);            /* H-STAR: function retained, extended internally */
 
     if (g_debug && me == 0) {
         printf("[DEBUG] npes=%d, leaf_size=%d, K=%d, levels=%d, num_groups[0]=%d\n",
                npes, G_LEAF, K, LEVELS, NUM_GROUPS0);
         fflush(stdout);
     }
 
     /* Run H-STAR termination */
     run_hstar_termination();
 
     shmem_finalize();
     return 0;
 }
 