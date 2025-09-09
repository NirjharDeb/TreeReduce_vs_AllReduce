/* treedone_bench.c
 * OpenSHMEM benchmark: non-collective tree termination (Youssef-style).
 */

 #define _POSIX_C_SOURCE 199309L

 #include <shmem.h>
 #include <assert.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <time.h>
 
 /* timing */
 static inline double now_sec(void) {
     struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
     return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
 }
 
 typedef struct {
     /* topology */
     int me, np, fanout;
     int parent, first_child, last_child, num_children;
 
     /* symmetric state */
     int *LOCAL_DONE;   /* 0 or -1 */
     int *GLOBAL_DONE;  /* 0 or -1 */
     int *child_vals;   /* children set slots to 1 */
 } tree_done_t;
 
 static inline int slot_index(int child, int parent, int fanout) {
     return child - (fanout * parent + 1);
 }
 
 static void build_topology(tree_done_t *td) {
     td->parent = (td->me == 0) ? -1 : (td->me - 1) / td->fanout;
     td->first_child = td->fanout * td->me + 1;
     td->last_child  = td->first_child + td->fanout - 1;
     if (td->first_child >= td->np) {
         td->num_children = 0;
         td->first_child = td->last_child = -1;
     } else {
         if (td->last_child >= td->np) td->last_child = td->np - 1;
         td->num_children = td->last_child - td->first_child + 1;
     }
 }
 
 /* init / teardown */
 static void treedone_init(tree_done_t *td, int fanout) {
     td->me = shmem_my_pe();
     td->np = shmem_n_pes();
     td->fanout = (fanout < 2 ? 2 : fanout);
     build_topology(td);
 
     td->LOCAL_DONE  = shmem_malloc(sizeof(int));
     td->GLOBAL_DONE = shmem_malloc(sizeof(int));
     td->child_vals  = shmem_malloc((size_t)td->fanout * sizeof(int));
 
     if (!td->LOCAL_DONE || !td->GLOBAL_DONE || !td->child_vals) {
         if (td->me == 0) fprintf(stderr, "shmem_malloc failed\n");
         shmem_global_exit(1);
     }
 
     *td->LOCAL_DONE = 0;
     *td->GLOBAL_DONE = 0;
     for (int i = 0; i < td->fanout; ++i) td->child_vals[i] = 0;
 
     shmem_barrier_all();
 }
 
 static void treedone_finalize(tree_done_t *td) {
     shmem_barrier_all();
     if (td->child_vals)  shmem_free(td->child_vals);
     if (td->GLOBAL_DONE) shmem_free(td->GLOBAL_DONE);
     if (td->LOCAL_DONE)  shmem_free(td->LOCAL_DONE);
     memset(td, 0, sizeof(*td));
     shmem_barrier_all();
 }
 
 /* per-iter reset (outside timed region) */
 static void reset_for_round(tree_done_t *td) {
     *td->LOCAL_DONE = 0;
     *td->GLOBAL_DONE = 0;
     for (int i = 0; i < td->fanout; ++i) td->child_vals[i] = 0;
     shmem_quiet();
 }
 
 /* non-collective tree termination:
  * - each PE marks local done
  * - waits for its children to ping parent slot
  * - non-root pings its parent's slot once its subtree is done
  * - root, after collecting all, sets GLOBAL_DONE = -1
  * - broadcast of GLOBAL_DONE cascades down
  */
 static void treedone_async_tree(tree_done_t *td) {
     *td->LOCAL_DONE = -1;
     shmem_fence();
 
     /* wait on children (if any) */
     for (int i = 0; i < td->num_children; ++i)
         shmem_int_wait_until(&td->child_vals[i], SHMEM_CMP_EQ, 1);
 
     if (td->me != 0) {
         int p = td->parent;
         int slot = slot_index(td->me, p, td->fanout);
         assert(slot >= 0 && slot < td->fanout);
         shmem_int_p(&td->child_vals[slot], 1, p);
         shmem_quiet();
     } else {
         *td->GLOBAL_DONE = -1;
         shmem_quiet();
     }
 
     /* wait for global done and forward to children */
     shmem_int_wait_until(td->GLOBAL_DONE, SHMEM_CMP_EQ, -1);
     for (int i = 0; i < td->num_children; ++i) {
         int child = td->first_child + i;
         shmem_int_p(td->GLOBAL_DONE, -1, child);
     }
     shmem_quiet();
 }
 
 static void usage(const char *p) {
     if (shmem_my_pe() == 0)
         fprintf(stderr, "Usage: %s [--iters N] [--warmup W] [--fanout K] [--jitter_us J]\n", p);
 }
 
 int main(int argc, char **argv) {
     shmem_init();
 
     int me = shmem_my_pe();
     int np = shmem_n_pes();
 
     long iters  = 20000;
     long warmup = 200;
     int  fanout = 2;
     int  jitter_us = 0; /* optional stagger to show non-collective entry */
 
     for (int i = 1; i < argc; ++i) {
         if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = strtol(argv[++i], NULL, 10);
         else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = strtol(argv[++i], NULL, 10);
         else if (!strcmp(argv[i], "--fanout") && i + 1 < argc) fanout = atoi(argv[++i]);
         else if (!strcmp(argv[i], "--jitter_us") && i + 1 < argc) jitter_us = atoi(argv[++i]);
         else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); shmem_finalize(); return 1; }
     }
     if (iters <= 0 || warmup < 0 || fanout < 2) { usage(argv[0]); shmem_finalize(); return 1; }
 
     tree_done_t td;
     treedone_init(&td, fanout);
 
     if (me == 0) {
         printf("PEs=%d, iters=%ld, warmup=%ld, fanout=%d, jitter_us=%d\n", np, iters, warmup, fanout, jitter_us);
         fflush(stdout);
     }
     shmem_barrier_all();
 
     /* warmup */
     for (long k = 0; k < warmup; ++k) {
         reset_for_round(&td);
         shmem_barrier_all(); /* start of round (benchmarking only) */
 
         if (jitter_us > 0) {
             /* simple stagger: different offsets, no collectives */
             struct timespec ts;
             ts.tv_sec = 0;
             ts.tv_nsec = (long)((((unsigned)me * 1315423911u + (unsigned)k) % (unsigned)jitter_us) * 1000L);
             nanosleep(&ts, NULL);
         }
 
         treedone_async_tree(&td);
         shmem_barrier_all(); /* end of round (benchmarking only) */
     }
 
     /* time */
     shmem_barrier_all();
     double t0 = now_sec();
     for (long k = 0; k < iters; ++k) {
         reset_for_round(&td);
         shmem_barrier_all(); /* start of round (benchmarking only) */
 
         if (jitter_us > 0) {
             struct timespec ts;
             ts.tv_sec = 0;
             ts.tv_nsec = (long)((((unsigned)me * 2654435761u + (unsigned)k) % (unsigned)jitter_us) * 1000L);
             nanosleep(&ts, NULL);
         }
 
         treedone_async_tree(&td);
         shmem_barrier_all(); /* end of round (benchmarking only) */
     }
     double t1 = now_sec();
 
     if (me == 0) {
         double us = 1e6 * (t1 - t0) / (double)iters;
         printf("\nResults (avg per iteration, PE0 local timing):\n");
         printf("  Tree (non-collective) termination : %.2f us/iter\n", us);
         printf("  Note: barriers bracket rounds for timing only; the routine itself is non-collective.\n");
         fflush(stdout);
     }
 
     treedone_finalize(&td);
     shmem_finalize();
     return 0;
 }
 