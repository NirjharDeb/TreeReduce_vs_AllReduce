/* treedone_bench.c
 * OpenSHMEM benchmark: TreeReduce-based global termination vs. Allreduce (AND).
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
     int *LOCAL_DONE;            /* 0 or -1 */
     int *GLOBAL_DONE;           /* 0 or -1 */
     int *child_vals;            /* children set slots to 1 */
 
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
     td->and_src     = shmem_malloc(sizeof(int));
     td->and_dst     = shmem_malloc(sizeof(int));
     td->pWrk_int    = shmem_malloc(sizeof(int)  * SHMEM_REDUCE_MIN_WRKDATA_SIZE);
     td->pSync_red   = shmem_malloc(sizeof(long) * SHMEM_REDUCE_SYNC_SIZE);
 
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
 static void reset_for_tree_round(tree_done_t *td) {
     *td->LOCAL_DONE = 0;
     *td->GLOBAL_DONE = 0;
     for (int i = 0; i < td->fanout; ++i) td->child_vals[i] = 0;
 }
 static void reset_for_allreduce_round(tree_done_t *td) {
     *td->LOCAL_DONE = 0;
     *td->GLOBAL_DONE = 0;
     *td->and_src = 0;
     *td->and_dst = 0;
     for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) td->pSync_red[i] = SHMEM_SYNC_VALUE;
 }
 
 /* TreeReduce termination (no atomics): wait on child slots; k-ary broadcast */
 static void treedone_collective_tree(tree_done_t *td) {
     *td->LOCAL_DONE = -1;
 
     for (int i = 0; i < td->num_children; ++i)
         shmem_int_wait_until(&td->child_vals[i], SHMEM_CMP_EQ, 1);
 
     if (td->me != 0) {
         int p = td->parent;
         int slot = slot_index(td->me, p, td->fanout);
         assert(slot >= 0 && slot < td->fanout);
         shmem_int_p(&td->child_vals[slot], 1, p);
         shmem_fence();
 
         shmem_int_wait_until(td->GLOBAL_DONE, SHMEM_CMP_EQ, -1);
         for (int i = 0; i < td->num_children; ++i) {
             int child = td->first_child + i;
             shmem_int_p(td->GLOBAL_DONE, -1, child);
         }
         shmem_quiet();
     } else {
         *td->GLOBAL_DONE = -1;
         shmem_quiet();
         for (int i = 0; i < td->num_children; ++i) {
             int child = td->first_child + i;
             shmem_int_p(td->GLOBAL_DONE, -1, child);
         }
         shmem_quiet();
     }
 
     shmem_int_wait_until(td->GLOBAL_DONE, SHMEM_CMP_EQ, -1);
 }
 
 /* Allreduce termination (AND over 1 int) */
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
 
 static void usage(const char *p) {
     if (shmem_my_pe() == 0)
         fprintf(stderr, "Usage: %s [--iters N] [--warmup W] [--fanout K]\n", p);
 }
 
 int main(int argc, char **argv) {
     shmem_init();
 
     int me = shmem_my_pe();
     int np = shmem_n_pes();
 
     long iters  = 20000;
     long warmup = 200;
     int  fanout = 2;
 
     for (int i = 1; i < argc; ++i) {
         if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = strtol(argv[++i], NULL, 10);
         else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = strtol(argv[++i], NULL, 10);
         else if (!strcmp(argv[i], "--fanout") && i + 1 < argc) fanout = atoi(argv[++i]);
         else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); shmem_finalize(); return 1; }
     }
     if (iters <= 0 || warmup < 0 || fanout < 2) { usage(argv[0]); shmem_finalize(); return 1; }
 
     tree_done_t td;
     treedone_init(&td, fanout);
 
     if (me == 0) {
         printf("PEs=%d, iters=%ld, warmup=%ld, fanout=%d\n", np, iters, warmup, fanout);
         fflush(stdout);
     }
     shmem_barrier_all();
 
     /* warmup tree */
     for (long k = 0; k < warmup; ++k) {
         reset_for_tree_round(&td);
         shmem_barrier_all();
         treedone_collective_tree(&td);
         shmem_barrier_all();
     }
 
     /* time tree */
     shmem_barrier_all();
     double t0 = now_sec();
     for (long k = 0; k < iters; ++k) {
         reset_for_tree_round(&td);
         shmem_barrier_all();
         treedone_collective_tree(&td);
         shmem_barrier_all();
     }
     double t1 = now_sec();
 
     /* warmup allreduce */
     for (long k = 0; k < warmup; ++k) {
         reset_for_allreduce_round(&td);
         shmem_barrier_all();
         treedone_collective_allreduce(&td);
         shmem_barrier_all();
         for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) td.pSync_red[i] = SHMEM_SYNC_VALUE;
     }
 
     /* time allreduce */
     shmem_barrier_all();
     double t2 = now_sec();
     for (long k = 0; k < iters; ++k) {
         reset_for_allreduce_round(&td);
         shmem_barrier_all();
         treedone_collective_allreduce(&td);
         shmem_barrier_all();
         for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) td.pSync_red[i] = SHMEM_SYNC_VALUE;
     }
     double t3 = now_sec();
 
     if (me == 0) {
         double tree_us = 1e6 * (t1 - t0) / (double)iters;
         double allr_us = 1e6 * (t3 - t2) / (double)iters;
         double speedup = (tree_us > 0.0) ? (allr_us / tree_us) : 0.0;
 
         printf("\nResults (avg per iteration, PE0 local timing):\n");
         printf("  TreeReduce (k=%d) termination : %.2f us/iter\n", fanout, tree_us);
         printf("  Allreduce (AND) termination   : %.2f us/iter\n", allr_us);
         printf("  Rel. speed (Allreduce / Tree) : %.2fx  (>=1 ⇒ Tree faster)\n", speedup);
         fflush(stdout);
     }
 
     treedone_finalize(&td);
     shmem_finalize();
     return 0;
 }
 