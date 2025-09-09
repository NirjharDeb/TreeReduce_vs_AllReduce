/* treedone.c
 * Minimal OpenSHMEM TreeReduce-based global termination.
 *
 * Semantics:
 *  - Each PE calls treedone_collective_terminate() once when it is truly idle.
 *  - Parents wait for their children to report "done" (1 -> parent slot).
 *  - Non-root sends "done" to its parent; root, after all children report, sets
 *    GLOBAL_DONE = -1 on all PEs (simple broadcast).
 *  - Worker loops elsewhere should exit on (*GLOBAL_DONE == -1).
 *
 * Build (self-test): oshcc -O3 -std=c11 -DTREE_DONE_TEST treedone.c -o treedone
 * Run:               oshrun -n 8 ./treedone --fanout 2
 */

 #include <shmem.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <assert.h>
 #include <time.h>   /* nanosleep for the test driver */
 
 /* ---------------- Types & helpers ---------------- */
 
 typedef struct {
     /* Topology */
     int me, np;
     int fanout;                 /* k */
     int parent;                 /* -1 for root */
     int first_child, last_child;
     int num_children;
 
     /* Symmetric state (OpenSHMEM dynamic symmetric memory) */
     int *LOCAL_DONE;            /* 0 (not done) or -1 (done) */
     int *GLOBAL_DONE;           /* 0 (running) or -1 (terminate) */
     int *child_vals;            /* length = fanout; children write 1 into their slot */
 } tree_done_t;
 
 static inline int slot_index(int child, int parent, int fanout) {
     /* For heap-style children = {k*parent+1 .. k*parent+k}, slot in [0..fanout-1] */
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
 
 /* ---------------- Public API ---------------- */
 
 void treedone_init(tree_done_t *td, int fanout) {
     assert(td);
     td->me = shmem_my_pe();
     td->np = shmem_n_pes();
     td->fanout = (fanout < 2 ? 2 : fanout);
 
     build_topology(td);
 
     /* Symmetric allocations (collective) */
     td->LOCAL_DONE  = shmem_malloc(sizeof(int));
     td->GLOBAL_DONE = shmem_malloc(sizeof(int));
     td->child_vals  = shmem_malloc((size_t)td->fanout * sizeof(int));
 
     if (!td->LOCAL_DONE || !td->GLOBAL_DONE || !td->child_vals) {
         if (td->me == 0) fprintf(stderr, "treedone: shmem_malloc failed\n");
         shmem_global_exit(1);
     }
 
     *td->LOCAL_DONE  = 0;
     *td->GLOBAL_DONE = 0;
     for (int i = 0; i < td->fanout; ++i) td->child_vals[i] = 0;
 
     shmem_barrier_all(); /* ensure everyone is initialized */
 }
 
 void treedone_finalize(tree_done_t *td) {
     shmem_barrier_all();
     if (td->child_vals)  shmem_free(td->child_vals);
     if (td->GLOBAL_DONE) shmem_free(td->GLOBAL_DONE);
     if (td->LOCAL_DONE)  shmem_free(td->LOCAL_DONE);
     memset(td, 0, sizeof(*td));
     shmem_barrier_all();
 }
 
 int *treedone_global_flag(tree_done_t *td) { return td->GLOBAL_DONE; }
 
 /* Collective: call once when this PE has finished all work. */
 void treedone_collective_terminate(tree_done_t *td) {
     *td->LOCAL_DONE = -1;
 
     /* Wait for all children (if any) to write 1 into our child_vals slots [0..num_children-1]. */
     if (td->num_children > 0) {
         const int need = td->num_children;
         for (;;) {
             int seen = 0;
             for (int i = 0; i < need; ++i) {
                 if (td->child_vals[i] == 1) ++seen;
             }
             if (seen == need) break;
             /* polite progress; no heavy spin */
             shmem_fence();
         }
     }
 
     if (td->me != 0) {
         /* Inform parent that our entire subtree is done. */
         int p = td->parent;
         int slot = slot_index(td->me, p, td->fanout);
         assert(slot >= 0 && slot < td->fanout);
         shmem_int_p(&td->child_vals[slot], 1, p);
         shmem_fence();
     } else {
         /* Root: entire system is done -> set GLOBAL_DONE on all PEs (simple broadcast). */
         *td->GLOBAL_DONE = -1;   /* self */
         shmem_quiet();
         for (int pe = 0; pe < td->np; ++pe) {
             if (pe == td->me) continue;
             shmem_int_p(td->GLOBAL_DONE, -1, pe);
         }
         shmem_quiet();
     }
 
     /* Ensure caller returns only after local view sees the global flag. */
     while (*td->GLOBAL_DONE != -1) shmem_fence();
 }
 
 /* ---------------------- Self-test driver (optional) ---------------------- */
 #ifdef TREE_DONE_TEST
 
 static void sleep_ms(int ms) {
     struct timespec ts;
     ts.tv_sec  = ms / 1000;
     ts.tv_nsec = (long)(ms % 1000) * 1000000L;
     nanosleep(&ts, NULL);
 }
 
 int main(int argc, char **argv) {
     shmem_init();
     int me = shmem_my_pe(), np = shmem_n_pes();
 
     int fanout = 2;
     for (int i = 1; i < argc; ++i) {
         if (!strcmp(argv[i], "--fanout") && i + 1 < argc) fanout = atoi(argv[++i]);
     }
 
     tree_done_t td;
     treedone_init(&td, fanout);
 
     if (me == 0) printf("PEs=%d, fanout=%d\n", np, fanout);
     shmem_barrier_all();
 
     /* Stagger completion to simulate work finishing at different times. */
     int delay_ms = (np - me - 1) * 50;
     sleep_ms(delay_ms);
 
     if (me == 0) printf("PE %d: calling TreeReduce termination\n", me);
     treedone_collective_terminate(&td);
 
     /* Verify */
     if (*treedone_global_flag(&td) != -1) {
         printf("PE %d: ERROR, GLOBAL_DONE not set\n", me);
         shmem_global_exit(2);
     }
     if (me == 0) printf("All PEs observe GLOBAL_DONE = %d. Success.\n", *treedone_global_flag(&td));
 
     treedone_finalize(&td);
     shmem_finalize();
     return 0;
 }
 #endif
 