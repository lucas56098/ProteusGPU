# Configuration file for compilation options
# Makefile converts these to -D flags

################################################################
# setup
################################################################

#dim_2D                     # run in 2D mode
dim_3D                      # run in 3D mode

#CUDA                       # enable GPU mode (not implemented yet)
CPU_DEBUG                   # run on CPU only (mandatory for now)

USE_HDF5                    # HDF5 for IC and output (mandatory)

################################################################
# hydro
################################################################

MOVING_MESH                         # enable moving mesh hydrodynamics
_CELL_SHAPING_SPEED_=0.7            # mesh regularization speed fraction
_CELL_SHAPING_FACTOR_=0.2           # regularization threshold in cell radii

_GAMMA_EOS_=1.6666666666666667      # adiabatic index

#RIEMANN_HLL                        # use HLL riemann solver
RIEMANN_HLLC                        # use HLLC riemann solver

################################################################
# parallelization
################################################################

USE_OPENMP                  # enable multithreading on CPU
_VORO_BLOCK_SIZE_=16        # threads for voronoi cells
_OMP_HYDRO_THREADS_=16      # threads for hydro solver

################################################################
# compile time memory constraints
################################################################

_K_=190                     # number of nearest neighbors (KNN)
_MAX_P_=64                  # max number of clipping planes per Voronoi cell
_MAX_T_=96                  # max number of triangles per Voronoi cell
_FACE_CAPACITY_MULT_=17     # max face array entries allocated per cell

################################################################
# experimental / debug
################################################################

#DEBUG_MODE                 # verbose printout
#ENABLE_PROFILING            # profiling of main routines
#SAVE_MEMORY                 # float instead of double for selected variables