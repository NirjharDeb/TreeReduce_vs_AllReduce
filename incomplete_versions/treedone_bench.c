/* treedone_bench.c
 * OpenSHMEM benchmark: non-collective tree (Youssef-style) vs. Allreduce (AND).
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
 
     /* allreduce workspace */
     int  *and_src, *and_dst;
     int  *pWrk_int;             /* SHMEM_REDUCE_MIN_WRKDATA_SIZE */
     long *pSync_red;            /* SHMEM_REDUCE_SYNC_SIZE */
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
 
     td->and_src   = shmem_malloc(sizeof(int));
     td->and_dst   = shmem_malloc(sizeof(int));
     td->pWrk_int  = shmem_malloc(sizeof(int)  * SHMEM_REDUCE_MIN_WRKDATA_SIZE);
     td->pSync_red = shmem_malloc(sizeof(long) * SHMEM_REDUCE_SYNC_SIZE);
 
     if (!td->LOCAL_DONE || !td->GLOBAL_DONE || !td->child_vals ||
         !td->and_src || !td->and_dst || !td->pWrk_int || !td->pSync_red) {
         if (td->me == 0) fprintf(stderr, "shmem_malloc failed\n");
         shmem_global_exit(1);
     }
 
     *td->LOCAL_DONE = 0;
     *td->GLOBAL_DONE = 0;
     for (int i = 0; i < td->fanout; ++i) td->child_vals[i] = 0;
     for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) td->pSync_red[i] = SHMEM_SYNC_VALUE;
 
     shmem_barrier_all();
 }
 
 static void treedone_finalize(tree_done_t *td) {
     shmem_barrier_all();
     if (td->pSync_red)  shmem_free(td->pSync_red);
     if (td->pWrk_int)   shmem_free(td->pWrk_int);
     if (td->and_dst)    shmem_free(td->and_dst);
     if (td->and_src)    shmem_free(td->and_src);
     if (td->child_vals) shmem_free(td->child_vals);
     if (td->GLOBAL_DONE) shmem_free(td->GLOBAL_DONE);
     if (td->LOCAL_DONE)  shmem_free(td->LOCAL_DONE);
     memset(td, 0, sizeof(*td));
     shmem_barrier_all();
 }
 
 /* per-iter reset (outside timed region) */
 static void reset_tree_round(tree_done_t *td) {
     *td->LOCAL_DONE = 0;
     *td->GLOBAL_DONE = 0;
     for (int i = 0; i < td->fanout; ++i) td->child_vals[i] = 0;
     shmem_quiet();
 }
 static void reset_allreduce_round(tree_done_t *td) {
     *td->LOCAL_DONE = 0;
     *td->GLOBAL_DONE = 0;
     *td->and_src = 0;
     *td->and_dst = 0;
     for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) td->pSync_red[i] = SHMEM_SYNC_VALUE;
     shmem_quiet();
 }
 
 /* non-collective tree */
 static void treedone_async_tree(tree_done_t *td) {
     *td->LOCAL_DONE = -1;
     shmem_fence();
 
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
 
     shmem_int_wait_until(td->GLOBAL_DONE, SHMEM_CMP_EQ, -1);
     for (int i = 0; i < td->num_children; ++i) {
         int child = td->first_child + i;
         shmem_int_p(td->GLOBAL_DONE, -1, child);
     }
     shmem_quiet();
 }
 
 /* collective allreduce (for comparison only) */
 static void treedone_collective_allreduce(tree_done_t *td) {
     *td->LOCAL_DONE = -1;
     *td->and_src = 1;
 
     shmem_int_and_to_all(
         td->and_dst, td->and_src, 1,
         0, 0, td->np,
         td->pWrk_int, td->pSync_red
     );
 
     if (*td->and_dst == 1) *td->GLOBAL_DONE = -1;
     shmem_int_wait_until(td->GLOBAL_DONE, SHMEM_CMP_EQ, -1);
 }
 
 /* jitter helper */
 static inline void nanosleep_us(unsigned usec) {
     if (!usec) return;
     struct timespec ts; ts.tv_sec = usec / 1000000u;
     ts.tv_nsec = (long)(usec % 1000000u) * 1000L;
     nanosleep(&ts, NULL);
 }
 
 static void usage(const char *p) {
     if (shmem_my_pe() == 0)
         fprintf(stderr, "Usage: %s [--iters N] [--warmup W] [--fanout K] [--bench both|tree|allreduce] [--jitter_us J]\n", p);
 }
 
 int main(int argc, char **argv) {
     shmem_init();
 
     int me = shmem_my_pe();
     int np = shmem_n_pes();
 
     long iters  = 4096;
     long warmup = 200;
     int  fanout = 2;
     int  jitter_us = 0;
     enum { BENCH_BOTH, BENCH_TREE, BENCH_ALLR } mode = BENCH_BOTH;
 
     for (int i = 1; i < argc; ++i) {
         if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = strtol(argv[++i], NULL, 10);
         else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = strtol(argv[++i], NULL, 10);
         else if (!strcmp(argv[i], "--fanout") && i + 1 < argc) fanout = atoi(argv[++i]);
         else if (!strcmp(argv[i], "--jitter_us") && i + 1 < argc) jitter_us = atoi(argv[++i]);
         else if (!strcmp(argv[i], "--bench") && i + 1 < argc) {
             const char *b = argv[++i];
             if (!strcmp(b, "tree")) mode = BENCH_TREE;
             else if (!strcmp(b, "allreduce")) mode = BENCH_ALLR;
             else mode = BENCH_BOTH;
         } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); shmem_finalize(); return 1; }
     }
     if (iters <= 0 || warmup < 0 || fanout < 2) { usage(argv[0]); shmem_finalize(); return 1; }
 
     tree_done_t td;
     treedone_init(&td, fanout);
 
     if (me == 0) {
         const char *m = (mode == BENCH_BOTH ? "both" : (mode == BENCH_TREE ? "tree" : "allreduce"));
         printf("PEs=%d, iters=%ld, warmup=%ld, fanout=%d, bench=%s, jitter_us=%d\n",
                np, iters, warmup, fanout, m, jitter_us);
         fflush(stdout);
     }
     shmem_barrier_all();
 
     /* warmup: tree */
     if (mode == BENCH_BOTH || mode == BENCH_TREE) {
         for (long k = 0; k < warmup; ++k) {
             reset_tree_round(&td);
             shmem_barrier_all();
             if (jitter_us) nanosleep_us(((unsigned)me * 2654435761u + (unsigned)k) % (unsigned)jitter_us);
             treedone_async_tree(&td);
             shmem_barrier_all();
         }
     }
 
     /* time: tree */
     double t_tree0 = 0.0, t_tree1 = 0.0;
     if (mode == BENCH_BOTH || mode == BENCH_TREE) {
         shmem_barrier_all();
         t_tree0 = now_sec();
         for (long k = 0; k < iters; ++k) {
             reset_tree_round(&td);
             shmem_barrier_all();
             if (jitter_us) nanosleep_us(((unsigned)me * 1315423911u + (unsigned)k) % (unsigned)jitter_us);
             treedone_async_tree(&td);
             shmem_barrier_all();
         }
         t_tree1 = now_sec();
     }
 
     /* warmup: allreduce */
     if (mode == BENCH_BOTH || mode == BENCH_ALLR) {
         for (long k = 0; k < warmup; ++k) {
             reset_allreduce_round(&td);
             shmem_barrier_all();
             treedone_collective_allreduce(&td);
             shmem_barrier_all();
         }
     }
 
     /* time: allreduce */
     double t_allr0 = 0.0, t_allr1 = 0.0;
     if (mode == BENCH_BOTH || mode == BENCH_ALLR) {
         shmem_barrier_all();
         t_allr0 = now_sec();
         for (long k = 0; k < iters; ++k) {
             reset_allreduce_round(&td);
             shmem_barrier_all();
             treedone_collective_allreduce(&td);
             shmem_barrier_all();
         }
         t_allr1 = now_sec();
     }
 
     if (me == 0) {
         double tree_us = (mode == BENCH_ALLR) ? 0.0 : 1e6 * (t_tree1 - t_tree0) / (double)iters;
         double allr_us = (mode == BENCH_TREE) ? 0.0 : 1e6 * (t_allr1 - t_allr0) / (double)iters;
 
         printf("\nResults (avg per iteration, PE0 local timing):\n");
         if (mode == BENCH_BOTH || mode == BENCH_TREE)
             printf("  Tree (non-collective) termination : %.2f us/iter\n", tree_us);
         if (mode == BENCH_BOTH || mode == BENCH_ALLR)
             printf("  Allreduce (AND) termination       : %.2f us/iter\n", allr_us);
 
         if (mode == BENCH_BOTH) {
             double speedup = (tree_us > 0.0) ? (allr_us / tree_us) : 0.0;
             printf("  Rel. speed (Allreduce / Tree)     : %.2fx  (>=1 ⇒ Tree faster)\n", speedup);
         }
         fflush(stdout);
     }
 
     treedone_finalize(&td);
     shmem_finalize();
     return 0;
 }
 