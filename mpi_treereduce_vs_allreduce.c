// mpi_treereduce_vs_allreduce.c
// Compare a manual k-ary TreeReduce (+ down-broadcast) against MPI_Allreduce.
// Build: mpicc -O3 -march=native -std=c11 mpi_treereduce_vs_allreduce.c -o mpi_bench
// Run:   mpirun -np 8 --oversubscribe --bind-to none ./mpi_bench --iters 20000 --count 1 --checks

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static inline long value_for_iter(long k, int me) {
    return k + 1 + me; // make each iteration’s value change
}

static void usage_and_exit(const char *prog) {
    fprintf(stderr,
        "Usage: %s [--iters N] [--warmup W] [--count C] [--fanout K] [--checks]\n", prog);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

enum { TAG_REDUCE = 1001, TAG_BCAST = 1002 };

/* Plan describing my place in a k-ary heap tree rooted at rank 0. */
typedef struct {
    int me, np;
    int fanout;
    int parent;                 // -1 for root
    int first_child, last_child;
    int num_children;

    // per-iteration scratch (allocated once, reused)
    int   count;
    long *acc;                  // accumulator (size=count)
    long *tmp_all;              // receive buffers from children (size=num_children*count)
    MPI_Request *red_recvs;     // Irecv handles from children
    MPI_Request *bcast_sends;   // Isend handles to children (broadcast phase)
} TreePlan;

static void tree_plan_init(TreePlan *pl, int fanout, int count, MPI_Comm comm) {
    memset(pl, 0, sizeof(*pl));
    MPI_Comm_rank(comm, &pl->me);
    MPI_Comm_size(comm, &pl->np);

    pl->fanout = (fanout < 2 ? 2 : fanout);
    pl->count  = count;

    pl->parent = (pl->me == 0) ? -1 : (pl->me - 1) / pl->fanout;

    // heap-style children: {k*i+1 ... k*i+k}
    pl->first_child = pl->fanout * pl->me + 1;
    pl->last_child  = pl->first_child + pl->fanout - 1;
    if (pl->first_child >= pl->np) {
        pl->num_children = 0;
        pl->first_child = pl->last_child = -1;
    } else {
        if (pl->last_child >= pl->np) pl->last_child = pl->np - 1;
        pl->num_children = pl->last_child - pl->first_child + 1;
    }

    pl->acc = (long*)malloc((size_t)count * sizeof(long));
    if (!pl->acc) { perror("malloc acc"); MPI_Abort(comm, 2); }

    if (pl->num_children > 0) {
        size_t block = (size_t)count;
        pl->tmp_all      = (long*)malloc((size_t)pl->num_children * block * sizeof(long));
        pl->red_recvs    = (MPI_Request*)malloc((size_t)pl->num_children * sizeof(MPI_Request));
        pl->bcast_sends  = (MPI_Request*)malloc((size_t)pl->num_children * sizeof(MPI_Request));
        if (!pl->tmp_all || !pl->red_recvs || !pl->bcast_sends) {
            perror("malloc children scratch");
            MPI_Abort(comm, 2);
        }
    }
}

static void tree_plan_free(TreePlan *pl) {
    free(pl->bcast_sends);
    free(pl->red_recvs);
    free(pl->tmp_all);
    free(pl->acc);
    memset(pl, 0, sizeof(*pl));
}

/* k-ary TreeReduce (sum of longs) followed by a down-broadcast of the result.
   Nonblocking Irecv from children so all arrivals can overlap.
   A single blocking send to the parent. 
   A nonblocking Isend for the broadcast fan-out. 
*/
static void kary_tree_reduce_bcast_sum_long_nb(
    const long *sendbuf, long *recvbuf, const TreePlan *pl, MPI_Comm comm)
{
    const int count = pl->count;

    memcpy(pl->acc, sendbuf, (size_t)count * sizeof(long));

    // Upward reduce: gather from children, then accumulate into acc
    if (pl->num_children > 0) {
        for (int i = 0; i < pl->num_children; ++i) {
            int child = pl->first_child + i;
            long *dst = pl->tmp_all + (size_t)i * (size_t)count;
            MPI_Irecv(dst, count, MPI_LONG, child, TAG_REDUCE, comm, &pl->red_recvs[i]);
        }
        MPI_Waitall(pl->num_children, pl->red_recvs, MPI_STATUSES_IGNORE);

        for (int i = 0; i < pl->num_children; ++i) {
            long *src = pl->tmp_all + (size_t)i * (size_t)count;
            for (int j = 0; j < count; ++j) pl->acc[j] += src[j];
        }
    }

    // Non-root forwards upward once (blocking is fine—one parent)
    if (pl->me != 0) {
        MPI_Send(pl->acc, count, MPI_LONG, pl->parent, TAG_REDUCE, comm);
        // Then wait for the broadcast from parent
        MPI_Recv(pl->acc, count, MPI_LONG, pl->parent, TAG_BCAST, comm, MPI_STATUS_IGNORE);
    }
    // Root already has the final sum in pl->acc at this point.

    // Downward broadcast: push to each child
    if (pl->num_children > 0) {
        for (int i = 0; i < pl->num_children; ++i) {
            int child = pl->first_child + i;
            MPI_Isend(pl->acc, count, MPI_LONG, child, TAG_BCAST, comm, &pl->bcast_sends[i]);
        }
        MPI_Waitall(pl->num_children, pl->bcast_sends, MPI_STATUSES_IGNORE);
    }

    memcpy(recvbuf, pl->acc, (size_t)count * sizeof(long));
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int me, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    long iters  = 20000;
    long warmup = 100;
    int  checks = 0;
    int  count  = 1;
    int  fanout = 2;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--count") && i + 1 < argc) {
            count = (int)strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--fanout") && i + 1 < argc) {
            fanout = (int)strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--checks")) {
            checks = 1;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage_and_exit(argv[0]);
        }
    }
    if (iters <= 0 || warmup < 0 || count <= 0 || fanout <= 0) usage_and_exit(argv[0]);

    if (me == 0) {
        printf("MPI ranks=%d, iters=%ld, warmup=%ld, count=%d, fanout=%d, checks=%s\n",
               np, iters, warmup, count, fanout, checks ? "on" : "off");
        fflush(stdout);
    }

    long *my  = (long*)malloc((size_t)count * sizeof(long));
    long *out = (long*)malloc((size_t)count * sizeof(long));
    long *ref = (long*)malloc((size_t)count * sizeof(long));
    if (!my || !out || !ref) { if (me==0) perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 3); }

    TreePlan plan;
    tree_plan_init(&plan, fanout, count, MPI_COMM_WORLD);

    // Warmup --> optional correctness check vs MPI_Allreduce
    for (long k = 0; k < warmup; ++k) {
        for (int j = 0; j < count; ++j) my[j] = value_for_iter(k, me) + j;

        kary_tree_reduce_bcast_sum_long_nb(my, out, &plan, MPI_COMM_WORLD);

        if (checks) {
            MPI_Allreduce(my, ref, count, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
            for (int j = 0; j < count; ++j) {
                if (out[j] != ref[j]) {
                    fprintf(stderr,
                        "Warmup mismatch rank %d at elem %d: tree=%ld allreduce=%ld\n",
                        me, j, out[j], ref[j]);
                    MPI_Abort(MPI_COMM_WORLD, 4);
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Bench: manual TreeReduce (+Bcast)
    volatile long sink_tree = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (long k = 0; k < iters; ++k) {
        for (int j = 0; j < count; ++j) my[j] = value_for_iter(k, me) + j;
        kary_tree_reduce_bcast_sum_long_nb(my, out, &plan, MPI_COMM_WORLD);
        long sum_scalar = 0; for (int j = 0; j < count; ++j) sum_scalar += out[j];
        sink_tree += sum_scalar; // prevent over-optimization
    }
    double t1 = MPI_Wtime();

    // Bench: MPI_Allreduce
    volatile long sink_allr = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    for (long k = 0; k < iters; ++k) {
        for (int j = 0; j < count; ++j) my[j] = value_for_iter(k, me) + j;
        MPI_Allreduce(my, out, count, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        long sum_scalar = 0; for (int j = 0; j < count; ++j) sum_scalar += out[j];
        sink_allr += sum_scalar;
    }
    double t3 = MPI_Wtime();

    // Final correctness spot-check (cheap scalar compare)
    if (checks) {
        for (int j = 0; j < count; ++j) my[j] = value_for_iter(iters, me) + j;
        kary_tree_reduce_bcast_sum_long_nb(my, out, &plan, MPI_COMM_WORLD);
        MPI_Allreduce(my, ref, count, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        long t_scalar = 0, a_scalar = 0;
        for (int j = 0; j < count; ++j) { t_scalar += out[j]; a_scalar += ref[j]; }
        if (me == 0 && t_scalar != a_scalar) {
            fprintf(stderr, "[CHECK] mismatch: tree=%ld allreduce=%ld\n", t_scalar, a_scalar);
        }
    }

    if (me == 0) {
        double tree_us = 1e6 * (t1 - t0) / (double)iters;
        double allr_us = 1e6 * (t3 - t2) / (double)iters;
        printf("\nResults (avg per iteration):\n");
        printf("  TreeReduce (k=%d) + Bcast : %.2f us/iter\n", fanout, tree_us);
        printf("  MPI_Allreduce              : %.2f us/iter\n", allr_us);
        printf("  Rel. speed (Allreduce / Tree) : %.2fx  (>1 => Tree faster)\n",
               (tree_us > 0.0) ? (allr_us / tree_us) : 0.0);
        printf("  (accumulators) sink_tree=%ld sink_allreduce=%ld\n",
               (long)sink_tree, (long)sink_allr);
        fflush(stdout);
    }

    tree_plan_free(&plan);
    free(ref); free(out); free(my);
    MPI_Finalize();
    return 0;
}
