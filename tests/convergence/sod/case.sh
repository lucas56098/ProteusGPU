# Sod shock tube: rarefaction, contact and shock from a single jump.
#
# The solution is discontinuous, so second order is not on offer -- L1 converges at about
# first order and that is the correct answer. What this catches is a scheme that has
# stopped converging at all, or a limiter that has started smearing the shock.

CASE_DESC="Sod shock tube (discontinuous)"
CASE_DIM=2
CASE_RESOLUTIONS="32 64 128"
CASE_TIME_END=0.1     # short enough that the two fans stay clear of the box centre
CASE_IC="ics/create_sod.py --dimension 2 --perturbation 0"
CASE_MIN_ORDER=0.6    # measured 0.78 - 0.95; the floor is here to catch a scheme that
                      # has stopped converging, not to certify a rate
