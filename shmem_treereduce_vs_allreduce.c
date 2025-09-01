// shmem_treereduce_vs_allreduce.c
// Compare Naïve AllReduce via O(P) GETs vs. TreeReduce via shmem_*_to_all
// Build: oshcc -O3 shmem_treereduce_vs_allreduce.c -o shmem_bench
// Run:   oshrun -np 8 ./shmem_bench --iters 20000 --checks

#include <shmem.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static inline double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline long value_for_iter(long k, int me) {
    // Simple changing value so loops can't be optimized away
    return k + 1 + me;
}

static void usage_and_exit(const char *prog) {
    if (shmem_my_pe() == 0) {
        fprintf(stderr, "Usage: %s [--iters N] [--warmup W] [--checks]\n", prog);
    }
    shmem_finalize();
    exit(1);
}

int main(int argc, char **argv) {
    shmem_init();
    const int me   = shmem_my_pe();
    const int npes = shmem_n_pes();

    // -------------------- CLI --------------------
    long iters  = 10000;
    long warmup = 100;
    int  checks = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--checks")) {
            checks = 1;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage_and_exit(argv[0]);
        }
    }

    if (iters <= 0 || warmup < 0) usage_and_exit(argv[0]);
    if (me == 0) {
        printf("PEs=%d, iters=%ld, warmup=%ld, checks=%s\n",
               npes, iters, warmup, checks ? "on" : "off");
        fflush(stdout);
    }

    // -------------------- Symmetric data --------------------
    long *published  = shmem_malloc(sizeof(long));   // each PE "publishes" one value
    long *reduce_dst = shmem_malloc(sizeof(long));   // collective result lands here
    if (!published || !reduce_dst) {
        if (me == 0) fprintf(stderr, "shmem_malloc failed\n");
        shmem_global_exit(1);
    }

    // Workspace for collectives (per OpenSHMEM requirements)
    static long pSync[SHMEM_REDUCE_SYNC_SIZE];
    static long pWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) pSync[i] = SHMEM_SYNC_VALUE;
    shmem_barrier_all(); // ensure pSync is initialized on all PEs

    // -------------------- Warmup --------------------
    for (long k = 0; k < warmup; ++k) {
        *published = value_for_iter(k, me);
        shmem_barrier_all();

        // Naïve GET everyone
        volatile long sum_naive = 0;
        for (int pe = 0; pe < npes; ++pe) {
            long v = shmem_long_g(published, pe);
            sum_naive += v;
        }

        // Collective reduce (tree-ish under the hood)
        long src = *published;
        *reduce_dst = 0;
        shmem_long_sum_to_all(
            reduce_dst, &src, 1,
            /*PE_start*/0, /*logPE_stride*/0, /*PE_size*/npes,
            pWrk, pSync
        );
        shmem_barrier_all();

        if (checks && (sum_naive != *reduce_dst)) {
            fprintf(stderr, "Warmup mismatch on PE %d: naive=%ld tree=%ld\n",
                    me, sum_naive, *reduce_dst);
            shmem_global_exit(2);
        }
    }
    shmem_barrier_all();

    // -------------------- Benchmark: Naïve GET AllReduce --------------------
    double t0 = 0.0, t1 = 0.0;
    volatile long sink_naive = 0; // prevent optimization
    t0 = now_s();
    for (long k = 0; k < iters; ++k) {
        *published = value_for_iter(k, me);
        shmem_barrier_all();

        long sum = 0;
        for (int pe = 0; pe < npes; ++pe) {
            long v = shmem_long_g(published, pe);
            sum += v;
        }
        sink_naive += sum;
        shmem_barrier_all();
    }
    t1 = now_s();
    double naive_sec = t1 - t0;

    // -------------------- Benchmark: TreeReduce (collective) ----------------
    double t2 = 0.0, t3 = 0.0;
    volatile long sink_tree = 0;
    t2 = now_s();
    for (long k = 0; k < iters; ++k) {
        *published = value_for_iter(k, me);
        shmem_barrier_all();

        long src = *published;
        *reduce_dst = 0;
        shmem_long_sum_to_all(
            reduce_dst, &src, 1,
            /*PE_start*/0, /*logPE_stride*/0, /*PE_size*/npes,
            pWrk, pSync
        );
        sink_tree += *reduce_dst;
        shmem_barrier_all();
    }
    t3 = now_s();
    double tree_sec = t3 - t2;

    // -------------------- Optional correctness check -----------------------
    if (checks) {
        *published = value_for_iter(iters, me);
        shmem_barrier_all();
        long check_sum = 0;
        for (int pe = 0; pe < npes; ++pe) check_sum += shmem_long_g(published, pe);

        long src = *published, out = 0;
        shmem_long_sum_to_all(
            &out, &src, 1,
            /*PE_start*/0, /*logPE_stride*/0, /*PE_size*/npes,
            pWrk, pSync
        );
        shmem_barrier_all();
        if (me == 0 && check_sum != out) {
            fprintf(stderr, "[CHECK] mismatch: naive=%ld tree=%ld\n", check_sum, out);
        }
    }

    // -------------------- Report --------------------
    if (me == 0) {
        double naive_us = 1e6 * naive_sec / (double)iters;
        double tree_us  = 1e6 * tree_sec  / (double)iters;
        printf("\nResults (avg per iteration):\n");
        printf("  Naïve AllReduce via GETs : %.2f us/iter\n", naive_us);
        printf("  TreeReduce (sum_to_all)  : %.2f us/iter\n", tree_us);
        printf("  Speedup (naive / tree)   : %.2fx\n", naive_us / tree_us);
        printf("  (accumulators) sink_naive=%ld sink_tree=%ld\n",
               (long)sink_naive, (long)sink_tree);
        fflush(stdout);
    }

    shmem_free(published);
    shmem_free(reduce_dst);
    shmem_finalize();
    return 0;
}
