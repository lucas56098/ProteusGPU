# Linear sound wave on a periodic box.
#
# The solution is smooth, so this is the case that should show the scheme's formal
# second order. If anything else in this directory regresses to first order, look here
# first: a smooth problem failing means the reconstruction or the time integrator, not
# the limiter.

CASE_DESC="linear sound wave (smooth)"
CASE_DIM=2
CASE_RESOLUTIONS="32 64 128"
CASE_TIME_END=1.0     # one period: c_s = 1 and the wavelength is the box
CASE_IC="ics/create_acoustic_wave.py --dimension 2 --perturbation 0"
CASE_MIN_ORDER=1.8    # measured 1.95 - 2.04, static and moving
