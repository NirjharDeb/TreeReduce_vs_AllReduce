/* compile (examples):
 *   oshcc -O3 -std=c11 -o global_done_original global_done_original.c
 * run:
 *   oshrun -n 4 ./global_done_original
 */

 #include <shmem.h>
 #include <stdio.h>
 #include <stdlib.h>
 
 /* Symmetric state */
 static int *LOCAL_DONE;  /* allocated with shmem_malloc */
 
 static void global_done(void) {
     int me = shmem_my_pe();
     /* Optional: print which PE triggered termination */
     if (me == 0) {
         /* Keep output minimal to avoid interleaving noise */
         printf("global_done() invoked by PE %d\n", me);
         fflush(stdout);
     }
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
 
     /* fetch LOCAL_DONE on all PEs */
     for (int pe_id = 0; pe_id < shmem_n_pes(); pe_id++) {
         if (pe_id == shmem_my_pe()) {
             local_done_value = *LOCAL_DONE;
         } else {
             local_done_value = shmem_int_g(LOCAL_DONE, pe_id);
         }
         if (!local_done_value) { break; }
         global_done_flag += local_done_value;
     }
 
     /* if all LOCAL_DONE == (-1)*shmem_n_pes then can (safely) invoke global termination */
     if (global_done_flag == (-1 * shmem_n_pes())) {
         global_done();
     }
 }
 
 int main(int argc, char **argv) {
     shmem_init();
 
     /* Allocate symmetric memory for LOCAL_DONE and initialize to 0 */
     LOCAL_DONE = shmem_malloc(sizeof(int));
     if (!LOCAL_DONE) {
         /* If allocation fails, try to exit cleanly */
         shmem_global_exit(1);
     }
     *LOCAL_DONE = 0;
 
     /* Each PE calls initiate_global_done(); whichever PE finds all done will terminate all */
     initiate_global_done();
 
     /* If no PE reached global_done() in this call, just finalize normally. */
     shmem_finalize();
     return 0;
 }
 