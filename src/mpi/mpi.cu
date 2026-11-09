// implements the communication between ranks (mpi.h)

#include "mpi/mpi.h"

#ifdef USE_MPI
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mpi {

    namespace {

        // the comm world, set once at init
        struct World {
            int rank      = 0;
            int nranks    = 1;
            int node_rank = 0;
            int node_size = 1;
#ifdef USE_MPI
            MPI_Comm comm = MPI_COMM_NULL;
#endif
        };
        World world;

        enum class Operation { min, max, sum };

        template <typename Number> Number combined_over_ranks(Number value, Operation operation);

    } // namespace

    void init(int* argc, char*** argv) {
#ifdef USE_MPI
        MPI_Init(argc, argv);
        MPI_Comm_dup(MPI_COMM_WORLD, &world.comm);
        MPI_Comm_rank(world.comm, &world.rank);
        MPI_Comm_size(world.comm, &world.nranks);

        // the ranks that share this node, and the place of this rank among them
        MPI_Comm node;
        MPI_Comm_split_type(world.comm, MPI_COMM_TYPE_SHARED, world.rank, MPI_INFO_NULL, &node);
        MPI_Comm_rank(node, &world.node_rank);
        MPI_Comm_size(node, &world.node_size);
        MPI_Comm_free(&node);
#else
        (void)argc;
        (void)argv;
#endif
    }

    void finalize() {
#ifdef USE_MPI
        MPI_Comm_free(&world.comm);
        MPI_Finalize();
#endif
    }

#ifdef USE_MPI
    MPI_Comm communicator() {
        return world.comm;
    }
#endif

    int rank() {
        return world.rank;
    }
    int nranks() {
        return world.nranks;
    }
    bool is_root() {
        return world.rank == 0;
    }
    int node_rank() {
        return world.node_rank;
    }
    int node_size() {
        return world.node_size;
    }

    std::vector<std::string> about_library() {
        std::vector<std::string> lines;
#ifdef USE_MPI
#ifdef CUDA
#ifdef GPU_AWARE_MPI
        lines.push_back("GPU_AWARE_MPI is set");
#else
        lines.push_back("GPU_AWARE_MPI not set");
#endif
#ifdef MPIX_CUDA_AWARE_SUPPORT
        lines.push_back("MPI MPIX_Query_cuda_support() = " + std::to_string(MPIX_Query_cuda_support()));
#endif
        if (const char* enabled = std::getenv("MPICH_GPU_SUPPORT_ENABLED")) {
            lines.push_back(std::string("MPI MPICH_GPU_SUPPORT_ENABLED = ") + enabled);
        }
#endif
        char version[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
        int  length                                  = 0;
        if (MPI_Get_library_version(version, &length) == MPI_SUCCESS) {
            lines.push_back(std::string("MPI library = ") + version);
        }
#endif
        return lines;
    }

    void stop_all_ranks() {
        std::fflush(nullptr);
#ifdef USE_MPI
        int started = 0;
        MPI_Initialized(&started);
        if (started) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        std::exit(1);
    }

    int min_over_ranks(int value) {
        return combined_over_ranks(value, Operation::min);
    }
    long long min_over_ranks(long long value) {
        return combined_over_ranks(value, Operation::min);
    }
    double min_over_ranks(double value) {
        return combined_over_ranks(value, Operation::min);
    }
    int max_over_ranks(int value) {
        return combined_over_ranks(value, Operation::max);
    }
    long long max_over_ranks(long long value) {
        return combined_over_ranks(value, Operation::max);
    }
    double max_over_ranks(double value) {
        return combined_over_ranks(value, Operation::max);
    }
    long long sum_over_ranks(long long value) {
        return combined_over_ranks(value, Operation::sum);
    }

    void sum_over_ranks(long long* values, int n) {
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, values, n, MPI_LONG_LONG, MPI_SUM, world.comm);
#else
        (void)values;
        (void)n;
#endif
    }

    bool on_any_rank(bool value) {
        return max_over_ranks(value ? 1 : 0) == 1;
    }

    void gather_from_all_ranks(long long value, long long* value_of_rank) {
#ifdef USE_MPI
        MPI_Allgather(&value, 1, MPI_LONG_LONG, value_of_rank, 1, MPI_LONG_LONG, world.comm);
#else
        value_of_rank[0] = value;
#endif
    }

    std::vector<char> gather_items_from_all_ranks(const void* items, int n, int item_bytes) {
#ifdef USE_MPI

        std::vector<int> bytes_of_rank(world.nranks), first_byte(world.nranks, 0);

        const int bytes = n * item_bytes;

        MPI_Allgather(&bytes, 1, MPI_INT, bytes_of_rank.data(), 1, MPI_INT, world.comm);

        for (int r = 1; r < world.nranks; r++)
            first_byte[r] = first_byte[r - 1] + bytes_of_rank[r - 1];

        std::vector<char> all(first_byte[world.nranks - 1] + bytes_of_rank[world.nranks - 1]);

        MPI_Allgatherv(
            items, bytes, MPI_BYTE, all.data(), bytes_of_rank.data(), first_byte.data(), MPI_BYTE, world.comm);

        return all;
#else
        const char* first = static_cast<const char*>(items);
        return std::vector<char>(first, first + (size_t)n * item_bytes);
#endif
    }

    void exchange_counts(const int* send_count, int* receive_count) {
#ifdef USE_MPI
        MPI_Alltoall(send_count, 1, MPI_INT, receive_count, 1, MPI_INT, world.comm);
#else
        receive_count[0] = send_count[0];
#endif
    }

    void exchange_items(const void* send,
                        const int*  send_count,
                        void*       receive,
                        const int*  receive_count,
                        int         item_bytes) {
#ifdef USE_MPI
        std::vector<int> send_offset(world.nranks), receive_offset(world.nranks);
        for (int r = 1; r < world.nranks; r++) {
            send_offset[r]    = send_offset[r - 1] + send_count[r - 1];
            receive_offset[r] = receive_offset[r - 1] + receive_count[r - 1];
        }
        MPI_Datatype item;
        MPI_Type_contiguous(item_bytes, MPI_BYTE, &item);
        MPI_Type_commit(&item);
        MPI_Alltoallv(send,
                      send_count,
                      send_offset.data(),
                      item,
                      receive,
                      receive_count,
                      receive_offset.data(),
                      item,
                      world.comm);
        MPI_Type_free(&item);
#else
        (void)receive_count;
        if (send_count[0] > 0) std::memcpy(receive, send, (size_t)send_count[0] * item_bytes);
#endif
    }

    namespace {

#ifdef USE_MPI
        template <typename Number> MPI_Datatype mpi_type_of();
        template <> MPI_Datatype mpi_type_of<int>() {
            return MPI_INT;
        }
        template <> MPI_Datatype mpi_type_of<long long>() {
            return MPI_LONG_LONG;
        }
        template <> MPI_Datatype mpi_type_of<double>() {
            return MPI_DOUBLE;
        }

        MPI_Op mpi_operation_of(Operation operation) {
            switch (operation) {
            case Operation::min:
                return MPI_MIN;
            case Operation::max:
                return MPI_MAX;
            case Operation::sum:
                return MPI_SUM;
            }
            return MPI_OP_NULL;
        }
#endif

        template <typename Number> Number combined_over_ranks(Number value, Operation operation) {
#ifdef USE_MPI
            Number result;
            MPI_Allreduce(&value, &result, 1, mpi_type_of<Number>(), mpi_operation_of(operation), world.comm);
            return result;
#else
            (void)operation;
            return value;
#endif
        }

    } // namespace

} // namespace mpi
