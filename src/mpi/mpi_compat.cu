// implements the MPI helpers (mpi_compat.h)

#include "global/gpu_compat.h"
#include "mpi_compat.h"
#include "profiler/profiler.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef CPU_DEBUG
#include <cuda_runtime.h>
#endif

#if defined(USE_MPI) && defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#define PROTEUS_HAS_MPIX_QUERY_CUDA 1
#endif
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

    // MPI_Init, then one GPU per rank
    void init(int* argc, char*** argv) {
#ifdef USE_MPI
        int provided = 0;
        MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);

        s_initialized = true;

        MPI_Comm_rank(MPI_COMM_WORLD, &s_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &s_nranks);

        // rank inside its node, that is what picks the GPU
        MPI_Comm node_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
        int local_rank = 0;
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_size(node_comm, &s_node_local_size);
        MPI_Comm_free(&node_comm);

#ifndef CPU_DEBUG
        // ranks are spread over the GPUs of the node
        cudaError_t cerr = cudaGetDeviceCount(&s_gpus_per_node);
        if (cerr != cudaSuccess || s_gpus_per_node == 0) {
            exit_failure("[rank %d] no CUDA devices visible (err=%d)\n", s_rank, (int)cerr);
        }
        int dev = local_rank % s_gpus_per_node;
        cerr    = cudaSetDevice(dev);
        if (cerr != cudaSuccess) {
            exit_failure("[rank %d] cudaSetDevice(%d) failed (err=%d)\n", s_rank, dev, (int)cerr);
        }
        cudaDeviceSetLimit(cudaLimitStackSize, 8192);
#endif
#else
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

    // every fatal error ends here; under MPI it aborts, so no rank is left waiting
    void exit_failure(const char* fmt, ...) {
        std::va_list args;
        va_start(args, fmt);
        std::vfprintf(stderr, fmt, args);
        va_end(args);
        std::fflush(stderr);
        // close the profile log while HDF5 still works
        Profiler::abort_profile_log();
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

    // what the MPI library says about GPU pointers, printed once at startup
    void report_gpu_aware_mpi() {
        if (rank() != 0) return;
#ifdef USE_MPI
#ifndef CPU_DEBUG
#ifdef GPU_AWARE_MPI
        std::printf("BEGRUN: GPU_AWARE_MPI is set\n");
#else
        std::printf("BEGRUN: GPU_AWARE_MPI not set\n");
#endif

#ifdef PROTEUS_HAS_MPIX_QUERY_CUDA
#ifdef MPIX_CUDA_AWARE_SUPPORT
        std::printf("BEGRUN: MPI MPIX_Query_cuda_support() = %d\n", MPIX_Query_cuda_support());
#endif
#endif
        if (const char* v = std::getenv("MPICH_GPU_SUPPORT_ENABLED")) {
            std::printf("BEGRUN: MPI MPICH_GPU_SUPPORT_ENABLED = %s\n", v);
        }

#endif
        char version[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
        int  vlen                                    = 0;
        if (MPI_Get_library_version(version, &vlen) == MPI_SUCCESS) {
            std::printf("BEGRUN: MPI library = %s\n", version);
        }
        std::fflush(stdout);
#endif
    }

#if defined(USE_MPI) && !defined(CPU_DEBUG)
    static void sync_device() {
        cudaDeviceSynchronize();
    }
#endif

    // without GPU aware MPI the buffer has to sit on the host before it is sent
    void mpi_sync_before_send(const void* buf, size_t bytes) {
#if defined(CPU_DEBUG) || !defined(USE_MPI)
        (void)buf;
        (void)bytes;
        return;
#else
        sync_device();
#ifndef GPU_AWARE_MPI
        if (bytes > 0 && buf != nullptr) {
            gpu_prefetch_to_cpu(const_cast<void*>(buf), bytes);
            sync_device();
        }
#else
        (void)buf;
        (void)bytes;
#endif
#endif
    }

    // and goes back to the device after it arrived
    void mpi_sync_after_recv(void* buf, size_t bytes) {
#if defined(CPU_DEBUG) || !defined(USE_MPI)
        (void)buf;
        (void)bytes;
        return;
#else
#ifndef GPU_AWARE_MPI
        if (bytes > 0 && buf != nullptr) { gpu_prefetch_to_gpu(buf, bytes); }
#else
        (void)buf;
        (void)bytes;
#endif
#endif
    }

} // namespace proteus_mpi
