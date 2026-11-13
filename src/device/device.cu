// implements the functions of device that are no templates (device.h)

#include "device/device.h"

#include "logging/logging.h"
#include "mpi/mpi.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

namespace device {

    namespace {

        constexpr double BYTES_PER_GB = 1e9;

        // memory of this rank; on the GPU also the pool of temporary memory
        struct Memory {
            long limit_in_bytes  = LONG_MAX;
            long in_use_in_bytes = 0;
            long peak_in_bytes   = 0;
#ifdef CUDA
            cudaMemPool_t pool = nullptr;
#endif
        };
        Memory memory;

        void warn_if_limit_above(double bytes_of_rank, const char* what);
        void count_new_bytes(const char* name, long bytes);
        long bytes_held();
        void copy_block_of_this_rank(const void*        send,
                                     const mpi::Blocks& out,
                                     void*              receive,
                                     const mpi::Blocks& in,
                                     int                item_bytes);
#if defined(CUDA) && !defined(GPU_AWARE_MPI)
        void move_bytes(const void* memory, long bytes, int location_type, const char* what);
#endif

    } // namespace

    void init(int node_rank, int ranks_on_node, double memory_limit_in_gb) {
        memory.limit_in_bytes = (long)(memory_limit_in_gb * BYTES_PER_GB);
#ifdef CUDA
        int n_gpus = 0;
        stop_on_cuda_error(cudaGetDeviceCount(&n_gpus), "counting the GPUs of the node");
        if (n_gpus == 0) logging::stop_run("DEVICE: no GPU on this node");

        // with all GPUs of the node visible the node rank picks one; the scheduler may also show each rank only its own
        const int gpu = node_rank % n_gpus;
        stop_on_cuda_error(cudaSetDevice(gpu), "choosing the GPU");
        cudaDeviceProp properties;
        stop_on_cuda_error(cudaGetDeviceProperties(&properties, gpu), "reading the GPU properties");

        // the PCI address tells the physical GPUs apart, however many of them a rank sees
        const long long address =
            ((long long)properties.pciDomainID << 16) | (properties.pciBusID << 8) | properties.pciDeviceID;
        std::vector<long long> address_of_node_rank = mpi::gather_on_node(address);
        const int ranks_on_gpu = (int)std::count(address_of_node_rank.begin(), address_of_node_rank.end(), address);
        std::sort(address_of_node_rank.begin(), address_of_node_rank.end());
        const int gpus_used =
            (int)(std::unique(address_of_node_rank.begin(), address_of_node_rank.end()) - address_of_node_rank.begin());
        logging::print("BEGRUN: GPUs used per node = %d (%d ranks/node, %d GPUs visible to a rank)",
                       gpus_used,
                       ranks_on_node,
                       n_gpus);
        logging::print_from_this_rank("CUDA: rank %d on device %d (%s, PCI %04x:%02x:%02x), SM %d.%d, %d rank(s) on it",
                                      mpi::rank(),
                                      gpu,
                                      properties.name,
                                      properties.pciDomainID,
                                      properties.pciBusID,
                                      properties.pciDeviceID,
                                      properties.major,
                                      properties.minor,
                                      ranks_on_gpu);
        warn_if_limit_above((double)properties.totalGlobalMem / ranks_on_gpu, "its share of the GPU memory");

        // freed temporary memory stays in the pool up to the limit, so the next temporary can reuse it
        stop_on_cuda_error(cudaDeviceGetDefaultMemPool(&memory.pool, gpu), "finding the memory pool of the GPU");
        uint64_t kept = (uint64_t)memory.limit_in_bytes;
        stop_on_cuda_error(cudaMemPoolSetAttribute(memory.pool, cudaMemPoolAttrReleaseThreshold, &kept),
                           "keeping temporary memory in the pool");
#else
        (void)node_rank;
        const double node_ram = (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGE_SIZE);
        warn_if_limit_above(node_ram / ranks_on_node, "its share of the node RAM");
#endif
    }

    void* allocate_bytes(const char* name, long bytes) {
        if (bytes <= 0) return nullptr;
        count_new_bytes(name, bytes);

        void* memory_of_array = nullptr;
#ifdef CUDA
        stop_on_cuda_error(cudaMallocManaged(&memory_of_array, bytes), name);

        // the array prefers to stay on the GPU; the CPU reads it from there
        cudaMemLocation gpu = {};
        gpu.type            = cudaMemLocationTypeDevice;
        stop_on_cuda_error(cudaGetDevice(&gpu.id), "finding the GPU");
        stop_on_cuda_error(cudaMemAdvise(memory_of_array, bytes, cudaMemAdviseSetPreferredLocation, gpu), name);
#else
        memory_of_array = std::malloc(bytes);
        if (memory_of_array == nullptr) logging::stop_run("DEVICE: no memory left for array %s", name);
#endif
        return memory_of_array;
    }

    void free_bytes(void* memory_of_array, long bytes) {
        if (memory_of_array == nullptr) return;
#ifdef CUDA
        stop_on_cuda_error(cudaFree(memory_of_array), "freeing an array");
#else
        std::free(memory_of_array);
#endif
        memory.in_use_in_bytes -= bytes;
    }

    void copy_bytes(void* to, const void* from, long bytes) {
        if (bytes <= 0) return;
#ifdef CUDA
        const timers::Scope timer("GPU", timers::Type::gpu);
        stop_on_cuda_error(cudaMemcpy(to, from, bytes, cudaMemcpyDefault), "copying an array");
#else
        std::memcpy(to, from, bytes);
#endif
    }

    int n_threads() {
#ifdef USE_OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    long peak_memory_in_bytes() {
        return memory.peak_in_bytes;
    }

    void print_max_memory_usage() {
        constexpr double MIB = 1024.0 * 1024.0;
        const char*      tag = mpi::nranks() > 1 ? " (rank 0)" : "";

        // ru_maxrss is in KB on Linux
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        const double node_ram = (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGE_SIZE);
        logging::print("MAIN: maximum CPU memory used%s: %g MiB (%g MiB total)",
                       tag,
                       (double)usage.ru_maxrss * 1024.0 / MIB,
                       node_ram / MIB);
#ifdef CUDA
        size_t free_bytes = 0, total_bytes = 0;
        stop_on_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes), "reading the GPU memory");
        logging::print("MAIN: maximum GPU memory used%s: %g MiB (%g MiB total)",
                       tag,
                       (double)memory.peak_in_bytes / MIB,
                       (double)total_bytes / MIB);
#endif
    }

    // without GPU_AWARE_MPI, the items go to the host for MPI and back to the GPU after
    void exchange_items(const void*        send,
                        const mpi::Blocks& out,
                        void*              receive,
                        const mpi::Blocks& in,
                        int                item_bytes) {
        copy_block_of_this_rank(send, out, receive, in, item_bytes);
#if defined(CUDA) && !defined(GPU_AWARE_MPI)
        if (mpi::nranks() > 1) {
            move_bytes(send, (long)out.total * item_bytes, cudaMemLocationTypeHost, "items to send");
            move_bytes(receive, (long)in.total * item_bytes, cudaMemLocationTypeHost, "items to receive");
        }
        mpi::exchange_items(send, out, receive, in, item_bytes);
        if (mpi::nranks() > 1) {
            move_bytes(receive, (long)in.total * item_bytes, cudaMemLocationTypeDevice, "items received");
        }
#else
        mpi::exchange_items(send, out, receive, in, item_bytes);
#endif
    }

    // what the loops use (declared in loops.h)

    void wait_for_gpu(const char* what) {
#ifdef CUDA
        stop_on_cuda_error(cudaGetLastError(), what);
        stop_on_cuda_error(cudaDeviceSynchronize(), what);
#else
        (void)what;
#endif
    }

    // on the GPU from CUDA's pool, which is fast; there the whole pool counts, not each temporary
    void* allocate_temporary(long bytes) {
        void* temporary = nullptr;
#ifdef CUDA
        stop_on_cuda_error(cudaMallocAsync(&temporary, bytes, 0), "allocating temporary memory");
        count_new_bytes("temporary memory", 0);
#else
        count_new_bytes("temporary memory", bytes);
        temporary = std::malloc(bytes);
        if (temporary == nullptr) logging::stop_run("DEVICE: no memory left for temporary memory");
#endif
        return temporary;
    }

    // on the GPU back to the pool, where it still counts
    void free_temporary(void* temporary, long bytes) {
#ifdef CUDA
        (void)bytes;
        stop_on_cuda_error(cudaFreeAsync(temporary, 0), "freeing temporary memory");
#else
        std::free(temporary);
        memory.in_use_in_bytes -= bytes;
#endif
    }

#ifdef CUDA
    void stop_on_cuda_error(cudaError_t result, const char* what) {
        if (result != cudaSuccess) logging::stop_run("DEVICE: CUDA error at %s: %s", what, cudaGetErrorString(result));
    }
#endif

    namespace {

        void warn_if_limit_above(double bytes_of_rank, const char* what) {
            if (memory.limit_in_bytes > bytes_of_rank) {
                logging::print_from_this_rank(
                    "BEGRUN: Warning! rank %d: the memory limit of %.3f GB is more than %s, %.3f GB",
                    mpi::rank(),
                    memory.limit_in_bytes / BYTES_PER_GB,
                    what,
                    bytes_of_rank / BYTES_PER_GB);
            }
        }

        // counts new bytes and the peak; stops the run above the memory limit
        void count_new_bytes(const char* name, long bytes) {
            long held = bytes_held() + bytes;
#ifdef CUDA
            // first the pool gives back what it keeps
            if (held > memory.limit_in_bytes) {
                stop_on_cuda_error(cudaDeviceSynchronize(), "waiting for the GPU before the pool gives back memory");
                stop_on_cuda_error(cudaMemPoolTrimTo(memory.pool, 0), "giving back the memory the pool keeps");
                held = bytes_held() + bytes;
            }
#endif
            if (held > memory.limit_in_bytes) {
                logging::stop_run("DEVICE: %s takes this rank to %.3f GB, above its limit of %.3f GB",
                                  name,
                                  held / BYTES_PER_GB,
                                  memory.limit_in_bytes / BYTES_PER_GB);
            }
            memory.in_use_in_bytes += bytes;
            if (held > memory.peak_in_bytes) memory.peak_in_bytes = held;
        }

        // memory this rank holds; on the GPU everything in the pool too
        long bytes_held() {
#ifdef CUDA
            uint64_t in_pool = 0;
            stop_on_cuda_error(cudaMemPoolGetAttribute(memory.pool, cudaMemPoolAttrReservedMemCurrent, &in_pool),
                               "reading the memory the pool holds");
            return memory.in_use_in_bytes + (long)in_pool;
#else
            return memory.in_use_in_bytes;
#endif
        }

        // periodic copies of a rank's own cells are the only blocks a rank has with itself
        void copy_block_of_this_rank(const void*        send,
                                     const mpi::Blocks& out,
                                     void*              receive,
                                     const mpi::Blocks& in,
                                     int                item_bytes) {
            const int  here   = mpi::rank();
            const auto in_out = std::lower_bound(out.ranks.begin(), out.ranks.end(), here);
            const auto in_in  = std::lower_bound(in.ranks.begin(), in.ranks.end(), here);
            const bool in_both =
                in_out != out.ranks.end() && *in_out == here && in_in != in.ranks.end() && *in_in == here;
            if (!in_both) return;

            const long from = out.first[in_out - out.ranks.begin()];
            const long to   = in.first[in_in - in.ranks.begin()];
            copy_bytes(static_cast<char*>(receive) + to * item_bytes,
                       static_cast<const char*>(send) + from * item_bytes,
                       (long)out.counts[in_out - out.ranks.begin()] * item_bytes);
        }

#if defined(CUDA) && !defined(GPU_AWARE_MPI)
        // moves bytes to host or GPU in one go, not page by page as MPI touches them
        void move_bytes(const void* memory, long bytes, int location_type, const char* what) {
            if (bytes <= 0) return;
            const timers::Scope timer("GPU", timers::Type::gpu);
            cudaMemLocation     to = {};
            to.type                = (cudaMemLocationType)location_type;
            if (to.type == cudaMemLocationTypeDevice) stop_on_cuda_error(cudaGetDevice(&to.id), "finding the GPU");
            stop_on_cuda_error(cudaMemPrefetchAsync(memory, bytes, to, 0, 0), what);
            if (to.type == cudaMemLocationTypeHost) stop_on_cuda_error(cudaDeviceSynchronize(), what);
        }
#endif

    } // namespace

} // namespace device
