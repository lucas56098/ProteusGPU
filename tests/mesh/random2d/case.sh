# Uniform random seeds: cells of every shape, some of them through the CPU fallback.

CASE_DESC="2D random seeds, moving mesh"
CASE_DIM=2
CASE_FLAGS="MOVING_MESH"
CASE_N=32
CASE_TIME_END=0.02
CASE_IC="ics/create_sod.py --dimension 2 --mesh_mode random"
