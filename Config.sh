# Compile-time switches. Makefile turns them into -D flags, src/config/config.h turns those into constexpr flags.

################################################################
# setup
################################################################

dim_2D                  # run in 2D mode
#dim_3D                 # run in 3D mode

#CUDA                    # GPU mode, otherwise everything runs on CPU
#CUDA_FAST_MATH         # faster GPU math (no longer bitwise equal to CPU)

#USE_OPENMP              # multithreading on CPU
#USE_MPI                # multiprocessing (one GPU per rank)
#GPU_AWARE_MPI          # MPI can send/recv GPU memory directly (needs CUDA + USE_MPI + correct libraries)

#ENABLE_PROFILING       # hierarchical timers and profile.hdf5
#CUDA_PROFILING         # timers show up as named ranges in NsightSystems (needs CUDA + ENABLE_PROFILING)

#OUTPUT_DEBUG           # snapshots also hold mesh geometry and more timestep info

################################################################
# hydro options
################################################################

#INDIVIDUAL_TIMESTEPS               # activates hierarchical timestepping

#MOVING_MESH                        # mesh flows with the gas (otherwise static)
#REGULARIZE_MESH_DISTANCE=0.7       # seeds more than 0.2 cell radii off centroid get regularized at up
                                    # to this many sound speeds (default: 0.7)
#REGULARIZE_MESH_FACE_ANGLE=0.5     # cell with face angle above 2.25 gets regularized at up to this
                                    # many sound speeds (default: off)
#REGULARIZE_MESH_VOLUME=0.7         # cells smaller than the reference size (vol_ref_cell_size in param.txt, else
                                    # the mean cell) move to larger neighbours at up to this many signal speeds
                                    # (default: off)

#ADIABATIC_INDEX=1.4                # sets adiabatic index of the ideal gas (default: 5./3.)

#TREE_LEAF_POINTS=8                 # points in a leaf of the neighbour tree, 1 to 15 (default: 8)
#MESH_MIN_CELLS_ON_GPU=1000         # lists of fewer cells are built on the CPU, 0 for none (default: 1000)

################################################################
# astrophysics
################################################################

#ASTRO_PHYSICS          # main switch of this module. code units in cgs from param.txt

#NFW                    # static potential of a dark matter halo
#HERNQUIST              # static potential of the central galaxy
#SMBH                   # static potential of softened point source

#COOLING                # radiative cooling from a table, down to a temperature floor

#SF_FEEDBACK            # naive star formation feedback

#AGN_THERMAL            # chaotic cold accretion with thermal         
#AGN_KINETIC            # and kinetic mode

#LIMITERS               # limits on speed and temperature near the centre (removes energy and momentum!..)
