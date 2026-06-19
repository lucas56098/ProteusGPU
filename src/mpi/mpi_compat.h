#ifndef MPI_COMPAT_H
#define MPI_COMPAT_H
#pragma once

#include <cstddef>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace proteus_mpi {

    // init/finalize MPI communication
    void init(int* argc, char*** argv);
    void finalize();

    // getter functions
    int rank();
    int nranks();
    int node_local_size();
    int gpus_per_node();

    inline bool is_root() {
        return rank() == 0;
    }

    // exit MPI if failure
    void exit_failure(const char* fmt, ...) __attribute__((format(printf, 1, 2), noreturn));

    // print MPI library + GPU-aware path status on rank 0. Called once from begrun
    // after MPI_Init + cudaSetDevice. Reports the compile-time GPU_AWARE_MPI state,
    // OpenMPI's MPIX_Query_cuda_support() if available, and Cray MPICH's
    // MPICH_GPU_SUPPORT_ENABLED env var if set.
    void report_gpu_aware_mpi();

    // Synchronization barrier before an MPI send/recv whose buffer lives in
    // managed memory. Behavior:
    //   CPU_DEBUG:                    no-op
    //   CUDA, no GPU_AWARE_MPI:       cudaDeviceSynchronize() + gpu_prefetch_to_cpu
    //   CUDA + GPU_AWARE_MPI:         cudaDeviceSynchronize() only (buffer stays device-resident)
    //
    // Call before MPI_Isend/MPI_Alltoallv on `buf` to ensure (a) any pack kernel
    // writes are visible to the NIC / host, and (b) the data sits where the MPI
    // library expects it. bytes may be 0 to skip the prefetch hint (still syncs).
    void mpi_sync_before_send(const void* buf, size_t bytes);

    // Symmetric: after MPI_Waitall / MPI_Alltoallv on `buf`. Behavior:
    //   CPU_DEBUG:                    no-op
    //   CUDA, no GPU_AWARE_MPI:       gpu_prefetch_to_gpu (next kernel needs it on device)
    //   CUDA + GPU_AWARE_MPI:         no-op (buffer already device-resident)
    void mpi_sync_after_recv(void* buf, size_t bytes);

} // namespace proteus_mpi

#endif // MPI_COMPAT_H
