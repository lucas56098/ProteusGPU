# Seeds on an exact lattice: every cell is degenerate, so the CPU builds them with the exact tests.

CASE_DESC="2D exact lattice, moving mesh"
CASE_DIM=2
CASE_FLAGS="MOVING_MESH"
CASE_N=32
CASE_TIME_END=0.02
CASE_IC="ics/create_acoustic_wave.py --dimension 2 --mesh_mode cartesian --perturbation 0"
