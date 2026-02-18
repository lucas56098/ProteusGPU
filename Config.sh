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

# Dry-run mode for CI/CD (exits after printing welcome banner)
# NOTE: turning this off requires a correct IC.hdf5 file...
DRY_RUN
