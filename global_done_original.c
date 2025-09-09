/* global_done_original.c
 *
 * Minimal standalone OpenSHMEM program implementing initiate_global_done()
 * with basic performance metrics logged by the detecting PE(s), plus
 * aggregated per-PE elapsed times (min/max/avg) printed before exit.
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
 static int    *LOCAL_DONE;   /* allocated with shmem_malloc */
 static double *ELAPSED_MS;   /* each PE writes its own elapsed (ms) */
 
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
     int me   = shmem_my_pe();
     int npes = shmem_n_pes();
 
     /* need extra layer with LOCAL_DONE because some apps require mbs to send messages
        within the "request" that they are serving, so this keeps the mb active to do that */
 
     *LOCAL_DONE = -1;
 
     /* record this PE's elapsed time (ms) at the moment it declares local done */
     *ELAPSED_MS = (now_sec() - g_start_time) * 1e3;
 
     int global_done_flag = 0;
     int local_done_value;
 
     /* metrics for this scan */
     int scanned = 0;
     int remote_gets = 0;
 
     /* fetch LOCAL_DONE on all PEs */
     for (int pe_id = 0; pe_id < npes; pe_id++) {
         if (pe_id == me) {
             local_done_value = *LOCAL_DONE;
         } else {
             local_done_value = shmem_int_g(LOCAL_DONE, pe_id);
             remote_gets++;
         }
         scanned++;
 
         if (!local_done_value) { break; }
         global_done_flag += local_done_value;
     }
 
     /* if all LOCAL_DONE == (-1)*npes then can (safely) invoke global termination */
     if (global_done_flag == (-1 * npes)) {
         /* Print lightweight scan metrics for this detecting PE */
         printf("PE %d detected all-done: scanned=%d, remote_gets=%d\n",
                me, scanned, remote_gets);
         fflush(stdout);
 
         /* Aggregate per-PE elapsed times (ms) before exit */
         double sum = 0.0, min = 0.0, max = 0.0, val = 0.0;
         for (int pe_id = 0; pe_id < npes; pe_id++) {
             if (pe_id == me) {
                 val = *ELAPSED_MS;
             } else {
                 val = shmem_double_g(ELAPSED_MS, pe_id);
             }
             if (pe_id == 0) { min = max = val; }
             if (val < min) min = val;
             if (val > max) max = val;
             sum += val;
         }
         double avg = sum / (double)npes;
         printf("Aggregated ELAPSED_MS across %d PEs: min=%.3f ms  avg=%.3f ms  max=%.3f ms\n",
                npes, min, avg, max);
         fflush(stdout);
 
         global_done();
     }
 }
 
 int main(int argc, char **argv) {
     shmem_init();
 
     /* record a (roughly) common start point per-PE */
     g_start_time = now_sec();
 
     /* Allocate symmetric memory */
     LOCAL_DONE = shmem_malloc(sizeof(int));
     ELAPSED_MS = shmem_malloc(sizeof(double));
     if (!LOCAL_DONE || !ELAPSED_MS) {
         shmem_global_exit(1);
     }
     *LOCAL_DONE = 0;
     *ELAPSED_MS = 0.0;
 
     /* Each PE calls initiate_global_done(); whichever PE finds all done will terminate all */
     initiate_global_done();
 
     /* If no PE reached global_done() in this call, just finalize normally. */
     shmem_finalize();
     return 0;
 }
 