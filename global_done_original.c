/* global_done_original.c
 *
 * Minimal standalone OpenSHMEM program implementing initiate_global_done()
 * with basic performance metrics and aggregated per-PE elapsed times.
 *
 * Runtime debug control via env var:
 *   GLOBAL_DONE_DEBUG=0 (default) -> suppress per-PE prints; single aggregate line
 *   GLOBAL_DONE_DEBUG=1           -> enable per-PE prints and aggregate per detector
 *
 * Compile:
 *   oshcc -O3 -std=c11 -o global_done_original global_done_original.c
 *
 * Run (example with 24 PEs on 1 node via Slurm):
 *   srun --mpi=pmix -N 1 -n 24 ./global_done_original
 *   GLOBAL_DONE_DEBUG=1 srun --mpi=pmix -N 1 -n 24 ./global_done_original
 */

 #define _POSIX_C_SOURCE 199309L  /* for clock_gettime */

 #include <shmem.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <time.h>
 
 /* -------- timing helper -------- */
 static inline double now_sec(void) {
     struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC, &ts);
     return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
 }
 
 /* Symmetric state */
 static int    *LOCAL_DONE;   /* -1 = done, 0 = not done (symmetric) */
 static double *ELAPSED_MS;   /* each PE writes its own elapsed time (ms) */
 static int    *AGG_PRINTED;  /* first-writer-wins flag at root (PE 0) */
 
 /* Globals */
 static double g_start_time = 0.0;
 static int    g_debug = 0;   /* 0 = quiet (default), 1 = verbose */
 static const int ROOT_PE = 0;
 
 static int env_debug_enabled(void) {
     const char *e = getenv("GLOBAL_DONE_DEBUG");
     if (!e) return 0;
     /* treat any nonzero, non-empty value as "on" */
     if (e[0] == '\0' || e[0] == '0') return 0;
     return 1;
 }
 
 static void maybe_print_global_done_invoked(void) {
     if (!g_debug) return;
     int me   = shmem_my_pe();
     int npes = shmem_n_pes();
     double elapsed_ms = (now_sec() - g_start_time) * 1e3;
     printf("global_done() invoked by PE %d after %.3f ms (npes=%d)\n",
            me, elapsed_ms, npes);
     fflush(stdout);
 }
 
 static void global_done(void) {
     /* Optional (debug only): per-PE print */
     maybe_print_global_done_invoked();
 
     /* terminate entire job step */
     shmem_global_exit(0);
 }
 
 /* Signals current PE termination and, if all PEs are done, invokes global termination */
 static void initiate_global_done(void) {
     int me   = shmem_my_pe();
     int npes = shmem_n_pes();
 
     /* Mark local done and record elapsed time */
     *LOCAL_DONE = -1;
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
         /* Debug-only per-PE detection print */
         if (g_debug) {
             printf("PE %d detected all-done: scanned=%d, remote_gets=%d\n",
                    me, scanned, remote_gets);
             fflush(stdout);
         }
 
         /* Aggregate per-PE elapsed times (ms) before exit */
         double sum = 0.0, min = 0.0, max = 0.0, val = 0.0;
         for (int pe_id = 0; pe_id < npes; pe_id++) {
             val = (pe_id == me) ? *ELAPSED_MS : shmem_double_g(ELAPSED_MS, pe_id);
             if (pe_id == 0) { min = max = val; }
             if (val < min) min = val;
             if (val > max) max = val;
             sum += val;
         }
         double avg = sum / (double)npes;
 
         int should_print_aggregate = 1;
         if (!g_debug) {
             /* Quiet mode: only the first PE (globally) prints the aggregate.
                Use atomic CAS on ROOT_PE's AGG_PRINTED symmetric int. */
             int old = shmem_int_atomic_compare_swap(AGG_PRINTED, 0, 1, ROOT_PE);
             should_print_aggregate = (old == 0);
         }
 
         if (should_print_aggregate) {
             printf("Aggregated ELAPSED_MS across %d PEs: min=%.3f ms  avg=%.3f ms  max=%.3f ms\n",
                    npes, min, avg, max);
             fflush(stdout);
         }
 
         global_done();
     }
 }
 
 int main(int argc, char **argv) {
     shmem_init();
 
     /* Runtime debug flag (from environment) */
     g_debug = env_debug_enabled();
 
     /* Startup barrier ONLY for timing alignment (no impact on logic) */
     shmem_barrier_all();
     g_start_time = now_sec();
 
     /* Allocate symmetric memory */
     LOCAL_DONE  = shmem_malloc(sizeof(int));
     ELAPSED_MS  = shmem_malloc(sizeof(double));
     AGG_PRINTED = shmem_malloc(sizeof(int));
     if (!LOCAL_DONE || !ELAPSED_MS || !AGG_PRINTED) {
         shmem_global_exit(1);
     }
 
     /* Initialize symmetric objects */
     *LOCAL_DONE  = 0;
     *ELAPSED_MS  = 0.0;
 
     /* Only the root's AGG_PRINTED governs the first-writer-wins behavior.
        Initialize to 0 on all PEs (root is authoritative for the atomic CAS). */
     *AGG_PRINTED = 0;
 
     /* Each PE calls initiate_global_done(); whichever detects will terminate all */
     initiate_global_done();
 
     /* If no PE reached global_done() in this call, just finalize normally. */
     shmem_finalize();
     return 0;
 }
 