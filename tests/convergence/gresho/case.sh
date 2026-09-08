# Gresho vortex: a rotating equilibrium the scheme has to hold still.
#
# Centrifugal force balances the pressure gradient exactly, so the solution is
# time-independent and the error is whatever the scheme adds. The velocity profile has
# kinks at r=0.2 and r=0.4, which is what keeps the order near one.
#
# This is the case that punishes a scheme for not being Galilean invariant -- a moving
# mesh should do noticeably better here than a static one at the same resolution.

CASE_DESC="Gresho vortex (steady, rotating)"
CASE_DIM=2
CASE_RESOLUTIONS="32 64 128"
CASE_TIME_END=1.0     # ~0.8 of a rotation at the peak of the profile
CASE_IC="ics/create_gresho.py --mesh_mode cartesian --perturbation 0"
CASE_MIN_ORDER=0.9    # measured 1.03 - 1.48
