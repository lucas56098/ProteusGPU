#include "global/gpu_compat.h"
#include "mpi_compat.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef CPU_DEBUG
#include <cuda_runtime.h>
#endif

// OpenMPI exposes MPIX_Query_cuda_support via <mpi-ext.h>. Cray MPICH does not
// ship that header — fall back to the MPICH_GPU_SUPPORT_ENABLED env var only.
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

    // ============================================================
    // GPU-aware MPI banner + sync helpers
    // ============================================================

    void report_gpu_aware_mpi() {
        if (rank() != 0) return;
#ifdef USE_MPI
#ifdef GPU_AWARE_MPI
        std::printf("BEGRUN: MPI GPU_AWARE_MPI is COMPILED IN — buffers passed to MPI are device pointers.\n");
#else
        std::printf("BEGRUN: MPI GPU_AWARE_MPI is OFF — managed-memory buffers staged through host before MPI.\n");
#endif

        // OpenMPI runtime probe — only available when <mpi-ext.h> is present.
#ifdef PROTEUS_HAS_MPIX_QUERY_CUDA
#ifdef MPIX_CUDA_AWARE_SUPPORT
        std::printf("BEGRUN: MPI MPIX_Query_cuda_support() = %d\n", MPIX_Query_cuda_support());
#endif
#endif
        // Cray MPICH env var — set by sites that have GPU-aware MPI enabled.
        if (const char* v = std::getenv("MPICH_GPU_SUPPORT_ENABLED")) {
            std::printf("BEGRUN: MPI MPICH_GPU_SUPPORT_ENABLED = %s\n", v);
        }

        char version[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
        int  vlen                                    = 0;
        if (MPI_Get_library_version(version, &vlen) == MPI_SUCCESS) {
            // Strip trailing newlines so the banner stays one line per fact.
            for (int i = vlen - 1; i >= 0 && (version[i] == '\n' || version[i] == '\r'); i--)
                version[i] = '\0';
            std::printf("BEGRUN: MPI library = %s\n", version);
        }
        std::fflush(stdout);
#else
        // no MPI compiled — nothing useful to report.
#endif
    }

#if defined(USE_MPI) && !defined(CPU_DEBUG)
    static void sync_device() {
        cudaDeviceSynchronize();
    }
#else
    static inline void sync_device() {}
#endif

    void mpi_sync_before_send(const void* buf, size_t bytes) {
#if defined(CPU_DEBUG) || !defined(USE_MPI)
        (void)buf;
        (void)bytes;
        return;
#else
        // Always sync: even with GPU_AWARE_MPI on, pack kernel writes must be
        // visible to the NIC DMA before the MPI call.
        sync_device();
#ifndef GPU_AWARE_MPI
        // Library expects host-resident pointer — pull the managed pages over.
        if (bytes > 0 && buf != nullptr) {
            gpu_prefetch_to_cpu(const_cast<void*>(buf), bytes);
            sync_device(); // make sure prefetch is done before MPI reads
        }
#else
        (void)bytes;
#endif
#endif
    }

    void mpi_sync_after_recv(void* buf, size_t bytes) {
#if defined(CPU_DEBUG) || !defined(USE_MPI)
        (void)buf;
        (void)bytes;
        return;
#else
#ifndef GPU_AWARE_MPI
        // Library wrote into host pages — push them back to the device so the
        // next compute kernel doesn't page-fault each cache line.
        if (bytes > 0 && buf != nullptr) { gpu_prefetch_to_gpu(buf, bytes); }
#else
        (void)buf;
        (void)bytes;
#endif
#endif
    }

} // namespace proteus_mpi
