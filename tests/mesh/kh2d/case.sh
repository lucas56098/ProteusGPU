# Perturbed lattice sheared by the flow, so the cells the moving mesh makes get checked.

CASE_DESC="2D Kelvin-Helmholtz, moving mesh"
CASE_DIM=2
CASE_FLAGS="MOVING_MESH"
CASE_N=32
CASE_TIME_END=0.2
CASE_IC="ics/create_kh.py --dimension 2"
