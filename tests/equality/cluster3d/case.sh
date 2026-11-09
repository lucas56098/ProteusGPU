# 3D cool-core cluster with every astrophysical source. The gas near the centre counts as cold and dense, so that the
# black hole and the star formation act from the start.

CASE_DESC="3D cluster with every astrophysical source"
CASE_DIM=3
CASE_N=12
CASE_TIME_END=0.004
CASE_IC="ics/create_cluster_3d.py --n_side_center 24 --refine_radius_kpc 100 --n_side_outer 0 --param /dev/null"
CASE_SWITCHES="ASTRO_PHYSICS NFW HERNQUIST SMBH COOLING SF_FEEDBACK AGN_THERMAL AGN_KINETIC LIMITERS"
CASE_PARAMETERS="UnitLength_in_cm = 2.46854206e+24
UnitMass_in_g = 1.42957424e+48
UnitVelocity_in_cm_per_s = 1.0e+08
M_NFW = 6.6e+14
c_NFW = 5.0
H0 = 75.1
M_BCG = 2.4e+11
R_BCG = 10.0
M_BH = 4.0e+08
smbh_softening = 0.05
cooling_table = $ROOT/ics/cooling_table_schure2009.txt
T_floor = 1.0e4
Gamma_SNIa = 3.0e-14
E_SNIa = 1.0e51
alpha_SNIa = 1.0e-19
eff_SF = 5.0e-5
n_SF = 1.0e-3
T_SF = 4.0e7
R_SF = 100.0
R_acc = 30.0
T_cold_acc = 4.0e7
t_acc = 5.0
eta_agn = 0.01
R_T = 30.0
f_T = 0.25
f_K = 0.75
R_jet = 20.0
h_jet = 30.0
L_jet = 10.0
v_cap = 0.05
R_lim = 50.0
T_max_lim = 1.0e9
v_cap_lim = 0.05"
