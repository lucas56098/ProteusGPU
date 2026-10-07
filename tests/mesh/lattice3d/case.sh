# Seeds on a lattice whose spacing is not a power of two: degenerate up to rounding, built on the CPU.

CASE_DESC="3D rounded lattice, moving mesh"
CASE_DIM=3
CASE_FLAGS="MOVING_MESH"
CASE_N=10
CASE_TIME_END=0.02
CASE_IC="ics/create_acoustic_wave.py --dimension 3 --mesh_mode cartesian --perturbation 0"
