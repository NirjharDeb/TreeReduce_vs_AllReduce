/* global_done_star.c
 *
 * STAR-based global termination for OpenSHMEM (no atomics).
 *
 * Model (0 = not done, -1 = done):
 * 1) Each PE writes -1 to its *group-local slot* at the group's anchor PE.
 * 2) Each group anchor waits until all members are -1, then marks that group
 *    as done at the root anchor (PE 0).
 * 3) The root waits until all groups are done, broadcasts a single global flag (-1)
 *    to every PE, and (optionally) prints aggregated timing.
 * 4) Everyone waits on the global flag, hits a final barrier, and exits cleanly.
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
 
 /* Root’s record that each group has finished: ROOT_GROUP_DONE[g] == -1 means group g is done. */
 static int    *ROOT_GROUP_DONE;            /* length: NUM_GROUPS0, authoritative at ROOT_PE */
 
 /* Global termination gate: set/broadcast to -1 by the root when global completion is proven. */
 static int    *GLOBAL_TERMINATION_READY;   /* each PE waits on its local copy */
 
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
 
 /* size (in PEs) of a group at level L (L=0 is leaf) */
 static inline int group_span_at_level(int leaf_size, int level) {
     return leaf_size << level; /* leaf_size * (2^level) */
 }
 
 /* canonical (static) group owner PE id for (level, group_idx) */
 static inline int static_group_owner_pe(int leaf_size, int level, int group_idx) {
     (void) level; /* STAR only uses level 0, but keep signature for clarity */
     return group_idx * group_span_at_level(leaf_size, /*level=*/0);
 }
 
 /* ---------- STAR allocation (leaf groups only) ---------- */
 static void allocate_star_flags(int npes) {
     /* How many groups at "leaf" granularity (no higher levels needed for STAR) */
     NUM_GROUPS0 = ceil_div(npes, G_LEAF);
 
     /* Per-group per-member flags at group anchors */
     GROUP_PE_DONE = shmem_malloc(sizeof(int*) * NUM_GROUPS0);
     if (!GROUP_PE_DONE) shmem_global_exit(1);
     for (int g = 0; g < NUM_GROUPS0; g++) {
         GROUP_PE_DONE[g] = shmem_malloc(sizeof(int) * G_LEAF);
         if (!GROUP_PE_DONE[g]) shmem_global_exit(1);
         for (int i = 0; i < G_LEAF; i++) GROUP_PE_DONE[g][i] = 0; /* 0 = not done */
     }
 
     /* Root’s per-group record */
     ROOT_GROUP_DONE = shmem_malloc(sizeof(int) * NUM_GROUPS0);
     if (!ROOT_GROUP_DONE) shmem_global_exit(1);
     for (int g = 0; g < NUM_GROUPS0; g++) ROOT_GROUP_DONE[g] = 0;
 
     /* Global termination flag */
     GLOBAL_TERMINATION_READY = shmem_malloc(sizeof(int));
     if (!GLOBAL_TERMINATION_READY) shmem_global_exit(1);
     *GLOBAL_TERMINATION_READY = 0;
 }
 
 /* ---------- STAR termination protocol ---------- */
 static void run_star_termination(void) {
     const int me   = shmem_my_pe();
     const int npes = shmem_n_pes();
 
     /* Compute my group and my index within that group */
     const int gidx = me / G_LEAF;
     const int idx  = me % G_LEAF;
 
     /* Group anchor is the canonical first PE of the group */
     const int owner = static_group_owner_pe(G_LEAF, /*level=*/0, gidx);
 
     /* 1) Each PE marks its own slot at the group owner to -1 and flushes */
     /* Record local completion time like initiate_global_done() */
     *LOCAL_DONE = -1;
     *ELAPSED_MS = (now_sec() - g_start_time) * 1e3;
 
     shmem_int_p(&GROUP_PE_DONE[gidx][idx], -1, owner);
     shmem_quiet();
 
     /* 2) If I'm the group anchor, wait for my group's actual members, then notify root */
     if (me == owner) {
         /* Determine actual group size (tail group may be smaller than G_LEAF) */
         int start = owner;
         int end   = start + G_LEAF;
         if (end > npes) end = npes;
         const int gsize = end - start;
 
         for (int i = 0; i < gsize; i++) {
             shmem_int_wait_until(&GROUP_PE_DONE[gidx][i], SHMEM_CMP_EQ, -1);
         }
 
         /* Record at ROOT that this group is finished */
         if (owner == ROOT_PE) {
             ROOT_GROUP_DONE[gidx] = -1; /* local store at root */
         } else {
             shmem_int_p(&ROOT_GROUP_DONE[gidx], -1, ROOT_PE);
             shmem_quiet();
         }
     }
 
     /* 3) Root waits for all groups, then broadcasts the global termination gate */
     if (me == ROOT_PE) {
         for (int g = 0; g < NUM_GROUPS0; g++) {
             shmem_int_wait_until(&ROOT_GROUP_DONE[g], SHMEM_CMP_EQ, -1);
         }
 
         /* Optional: print simple timing summary (per-PE times when they set LOCAL_DONE) */
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
 
         /* IMPORTANT: broadcast to every PE's local flag so their waits complete */
         for (int pe = 0; pe < npes; pe++) {
             if (pe == ROOT_PE) {
                 *GLOBAL_TERMINATION_READY = -1;                 /* local store for root */
             } else {
                 shmem_int_p(GLOBAL_TERMINATION_READY, -1, pe);  /* remote PUT to each PE */
             }
         }
         shmem_quiet();  /* ensure all PUTs are visible */
     }
 
     /* 4) Everyone waits for the global gate; then prove *everyone* saw it and is exiting */
     shmem_int_wait_until(GLOBAL_TERMINATION_READY, SHMEM_CMP_EQ, -1);
 
     /* Final collective proof: if this completes, every PE observed the gate */
     shmem_barrier_all();
 
     if (me == ROOT_PE) {
         printf("ALL_CLEAR: all %d PEs observed termination and reached the final barrier.\n", npes);
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
 
     /* Align start for timing; not required for logic */
     shmem_barrier_all();
     g_start_time = now_sec();
 
     /* Symmetric allocations (local bookkeeping + STAR flags) */
     LOCAL_DONE = shmem_malloc(sizeof(int));
     ELAPSED_MS = shmem_malloc(sizeof(double));
     if (!LOCAL_DONE || !ELAPSED_MS) shmem_global_exit(1);
 
     *LOCAL_DONE = 0;
     *ELAPSED_MS = 0.0;
 
     allocate_star_flags(npes);
 
     if (g_debug && me == 0) {
         printf("[DEBUG] npes=%d, group_size=%d, num_groups=%d\n",
                npes, G_LEAF, NUM_GROUPS0);
         fflush(stdout);
     }
 
     /* Run STAR termination */
     run_star_termination();
 
     shmem_finalize();
     return 0;
 }
 