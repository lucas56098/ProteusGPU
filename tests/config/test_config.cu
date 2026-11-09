// config: every constexpr flag matches its switch in Config.sh.

#include "config/config.h"
#include "doctest.h"

TEST_CASE("config::dim matches dim_2D or dim_3D") {
#ifdef dim_2D
    CHECK(config::dim == 2);
#endif
#ifdef dim_3D
    CHECK(config::dim == 3);
#endif
}

TEST_CASE("config::use_gpu matches CUDA") {
#ifdef CUDA
    CHECK(config::use_gpu);
#else
    CHECK_FALSE(config::use_gpu);
#endif
}

TEST_CASE("config::use_mpi matches USE_MPI") {
#ifdef USE_MPI
    CHECK(config::use_mpi);
#else
    CHECK_FALSE(config::use_mpi);
#endif
}

TEST_CASE("config::use_openmp matches USE_OPENMP") {
#ifdef USE_OPENMP
    CHECK(config::use_openmp);
#else
    CHECK_FALSE(config::use_openmp);
#endif
}
