# Configuration file for compilation options
# Makefile converts these to -D flags

# 2D or 3D version of the code (one must be defined)
#dim_2D
dim_3D

# GPU settings
#CUDA -- not yet implemented
#HIP -- not yet implemented
CPU_DEBUG # -- mandatory for now

# HDF5 for IC and output (currently mandatory)
USE_HDF5

# Debug
#DEBUG_MODE

# Output types
#WRITE_KNN_OUTPUT

# Verification (bruteforce KNN check)
#VERIFY

# Dry-run mode for CI/CD (exits after printing welcome banner)
# NOTE: turning this off requires a correct IC.hdf5 file...
DRY_RUN

# Compile-time constants for KNN and Voronoi
_K_=190                  # number of nearest neighbors
_KNN_BLOCK_SIZE_=64     # number of threads per block for KNN
_VORO_BLOCK_SIZE_=16    # number of threads per block for Voronoi
_MAX_P_=64              # max number of clipping planes per Voronoi cell
_MAX_T_=96              # max number of triangles per Voronoi cell
