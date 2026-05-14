#include "mpi_init.h"
#include <cstdio>
#include <cstdlib>
#ifndef CPU_DEBUG
#include <cuda_runtime.h>
#endif

namespace proteus_mpi {

#ifdef USE_MPI

static int  g_rank            = 0;
static int  g_nranks          = 1;
static int  g_node_local_size = 1;
static int  g_gpus_per_node   = 0;
static bool g_initialized     = false;

void init(int* argc, char*** argv) {
    int provided = 0;
    int ierr     = MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed (err=%d)\n", ierr);
        std::exit(EXIT_FAILURE);
    }
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI did not provide MPI_THREAD_FUNNELED (got %d)\n", provided);
        std::exit(EXIT_FAILURE);
    }
    g_initialized = true;

    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

    // node-local rank for GPU pinning
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &g_node_local_size);
    MPI_Comm_free(&node_comm);

#ifndef CPU_DEBUG
    cudaError_t cerr = cudaGetDeviceCount(&g_gpus_per_node);
    if (cerr != cudaSuccess || g_gpus_per_node == 0) {
        fprintf(stderr, "[rank %d] no CUDA devices visible (err=%d)\n", g_rank, (int)cerr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // rank N on this node maps to GPU (N % gpus_per_node)
    int dev = local_rank % g_gpus_per_node;
    cerr    = cudaSetDevice(dev);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "[rank %d] cudaSetDevice(%d) failed (err=%d)\n", g_rank, dev, (int)cerr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif
}

void finalize() {
    if (g_initialized) {
        MPI_Finalize();
        g_initialized = false;
    }
}

int rank()            { return g_rank; }
int nranks()          { return g_nranks; }
int node_local_size() { return g_node_local_size; }
int gpus_per_node()   { return g_gpus_per_node; }

#else  // !USE_MPI

static int g_gpus_per_node_no_mpi = -1;

void init(int* /*argc*/, char*** /*argv*/) {
#ifndef CPU_DEBUG
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
    g_gpus_per_node_no_mpi = n;
#else
    g_gpus_per_node_no_mpi = 0;
#endif
}
void finalize() {}
int  rank()            { return 0; }
int  nranks()          { return 1; }
int  node_local_size() { return 1; }
int  gpus_per_node()   { return g_gpus_per_node_no_mpi < 0 ? 0 : g_gpus_per_node_no_mpi; }

#endif

}  // namespace proteus_mpi
