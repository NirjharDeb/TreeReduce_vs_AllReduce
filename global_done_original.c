/* global_done_original.c
 *
 * Minimal standalone OpenSHMEM program implementing initiate_global_done()
 * with basic performance metrics logged by the detecting PE(s).
 *
 * Compile:
 *   oshcc -O3 -std=c11 -o global_done_original global_done_original.c
 *
 * Run (example with 24 PEs on 1 node via Slurm):
 *   srun --mpi=pmix -N 1 -n 24 ./global_done_original
 */

 #define _POSIX_C_SOURCE 199309L  /* for clock_gettime */

 #include <shmem.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>
 
 /* -------- timing helper -------- */
 static inline double now_sec(void) {
     struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC, &ts);
     return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
 }
 
 /* Symmetric state */
 static int *LOCAL_DONE;            /* allocated with shmem_malloc */
 
 /* per-PE metrics (local) */
 static double g_start_time = 0.0;
 
 static void global_done(void) {
     int me   = shmem_my_pe();
     int npes = shmem_n_pes();
     double elapsed_ms = (now_sec() - g_start_time) * 1e3;
 
     /* Minimal: which PE triggered and how long since start */
     printf("global_done() invoked by PE %d after %.3f ms (npes=%d)\n",
            me, elapsed_ms, npes);
     fflush(stdout);
 
     /* Terminate the entire program cleanly for all PEs */
     shmem_global_exit(0);
 }
 
 /* Signals current PE termination and, if all PEs are done, invokes global termination */
 static void initiate_global_done(void) {
     /* need extra layer with LOCAL_DONE because some apps require mbs to send messages
        within the "request" that they are serving, so this keeps the mb active to do that */
 
     *LOCAL_DONE = -1;
 
     int global_done_flag = 0;
     int local_done_value;
 
     /* metrics for this scan */
     int scanned = 0;
     int remote_gets = 0;
 
     /* fetch LOCAL_DONE on all PEs */
     for (int pe_id = 0; pe_id < shmem_n_pes(); pe_id++) {
         if (pe_id == shmem_my_pe()) {
             local_done_value = *LOCAL_DONE;
         } else {
             local_done_value = shmem_int_g(LOCAL_DONE, pe_id);
             remote_gets++;
         }
         scanned++;
 
         if (!local_done_value) { break; }
         global_done_flag += local_done_value;
     }
 
     /* if all LOCAL_DONE == (-1)*shmem_n_pes then can (safely) invoke global termination */
     if (global_done_flag == (-1 * shmem_n_pes())) {
         /* print lightweight scan metrics before aborting the step */
         printf("PE %d detected all-done: scanned=%d, remote_gets=%d\n",
                shmem_my_pe(), scanned, remote_gets);
         fflush(stdout);
 
         global_done();
     }
 }
 
 int main(int argc, char **argv) {
     shmem_init();
 
     /* record a common start point per-PE (not synchronized; adequate for wall-time) */
     g_start_time = now_sec();
 
     /* Allocate symmetric memory for LOCAL_DONE and initialize to 0 */
     LOCAL_DONE = shmem_malloc(sizeof(int));
     if (!LOCAL_DONE) {
         shmem_global_exit(1);
     }
     *LOCAL_DONE = 0;
 
     /* Each PE calls initiate_global_done(); whichever PE finds all done will terminate all */
     initiate_global_done();
 
     /* If no PE reached global_done() in this call, just finalize normally. */
     shmem_finalize();
     return 0;
 }
 