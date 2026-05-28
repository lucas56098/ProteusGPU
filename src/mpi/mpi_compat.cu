#include "mpi_compat.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#ifndef CPU_DEBUG
#include <cuda_runtime.h>
#endif

namespace proteus_mpi {

#ifdef USE_MPI
    static int  s_rank            = 0;
    static int  s_nranks          = 1;
    static int  s_node_local_size = 1;
    static int  s_gpus_per_node   = 0;
    static bool s_initialized     = false;

#else
    static int s_gpus_per_node_no_mpi = -1;
#endif

    void init(int* argc, char*** argv) {
#ifdef USE_MPI
        // MPI_THREAD_FUNNELED: MPI calls only from one thread
        int provided = 0;
        MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);

        s_initialized = true;

        MPI_Comm_rank(MPI_COMM_WORLD, &s_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &s_nranks);

        // node-local rank/size for GPU pinning
        MPI_Comm node_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
        int local_rank = 0;
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_size(node_comm, &s_node_local_size);
        MPI_Comm_free(&node_comm);

#ifndef CPU_DEBUG
        cudaError_t cerr = cudaGetDeviceCount(&s_gpus_per_node);
        if (cerr != cudaSuccess || s_gpus_per_node == 0) {
            exit_failure("[rank %d] no CUDA devices visible (err=%d)\n", s_rank, (int)cerr);
        }
        // rank N on this node maps to GPU (N % gpus_per_node)
        int dev = local_rank % s_gpus_per_node;
        cerr    = cudaSetDevice(dev);
        if (cerr != cudaSuccess) {
            exit_failure("[rank %d] cudaSetDevice(%d) failed (err=%d)\n", s_rank, dev, (int)cerr);
        }
#endif
#else // !USE_MPI
        (void)argc;
        (void)argv;
#ifndef CPU_DEBUG
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
        s_gpus_per_node_no_mpi = n;
#else
        s_gpus_per_node_no_mpi = 0;
#endif
#endif
    }

    void finalize() {
#ifdef USE_MPI
        if (s_initialized) {
            MPI_Finalize();
            s_initialized = false;
        }
#endif
    }

    void exit_failure(const char* fmt, ...) {
        std::va_list args;
        va_start(args, fmt);
        std::vfprintf(stderr, fmt, args);
        va_end(args);
        std::fflush(stderr);
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        std::exit(EXIT_FAILURE);
#endif
        __builtin_unreachable();
    }

    int rank() {
#ifdef USE_MPI
        return s_rank;
#else
        return 0;
#endif
    }

    int nranks() {
#ifdef USE_MPI
        return s_nranks;
#else
        return 1;
#endif
    }

    int node_local_size() {
#ifdef USE_MPI
        return s_node_local_size;
#else
        return 1;
#endif
    }

    int gpus_per_node() {
#ifdef USE_MPI
        return s_gpus_per_node;
#else
        return s_gpus_per_node_no_mpi < 0 ? 0 : s_gpus_per_node_no_mpi;
#endif
    }

} // namespace proteus_mpi
