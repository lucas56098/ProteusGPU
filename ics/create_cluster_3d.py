"""
3D idealised cool-core cluster in hydrostatic equilibrium.

Native geometry for the setup of Fournier et al. (magnetised cold filaments, their Sect. 2.3): NFW
+ Hernquist(BCG) + softened SMBH point mass (their Eqs. 3-5) closed by the ACCEPT entropy profile
K(r)=K0+K100*(r/100kpc)^alpha_K (Eqs. 6-7) at n_e_ref@r_ref. Same code-unit constants that
src/astro/gravity.cu::gravity_init builds, so the IC is in HSE against gravity_accel.

Symmetry-breaking velocity field is solenoidal in 3D (v = curl(A), 40 vector-Fourier modes,
lambda in [12.5, 50] kpc, inverse-parabolic power spectrum peaked at 25 kpc, sigma_v).

The mesh is a stack of jittered cartesian layers, each contributing the cells in one radial
band: nested factor-2 levels around the centre, a base layer, and a coarse outer shell. With
--smooth a single fine grid is rejection-thinned to a smoothly varying cell size instead.

Writes the IC hdf5 AND a matching param file so the unit system can never drift between them.

Runs in either mode:
  python create_cluster_3d.py --n 64
  mpirun -np 4 python create_cluster_3d.py --n 128
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from common import (
    Mesh,
    build_arg_parser,
    mpi_runtime,
    per_particle_signed,
    per_particle_uniform,
    resolve_filename,
    write_ic,
)

# cgs constants
G_CGS = 6.67430e-8
KPC = 3.085677581e21
MPC = 3.085677581e24
MSUN = 1.989e33
M_H = 1.67262192e-24
K_B = 1.380649e-16
KEV_ERG = 1.602176634e-9
KM_S = 1.0e5
X_H = 0.76
MU = 0.6                    # ionized ICM mean molecular weight
MU_E = 2.0 / (1.0 + X_H)    # mean mass per electron / m_H

# --- Table 1 (Fournier et al.) ---
M_NFW = 6.6e14      # Msun
C_NFW = 5.0
H0 = 75.1           # km/s/Mpc
M_BCG = 2.4e11      # Msun
R_BCG = 10.0        # kpc
M_BH = 4.0e8        # Msun
EPS_SMBH = 0.05     # kpc (Plummer softening)
K0 = 10.0           # keV cm^2
K100 = 150.0        # keV cm^2
ALPHA_K = 1.1
N_E_REF = 0.05      # cm^-3
R_REF = 10.0        # kpc


def build_gravity_code_units(UL, UM, UT):
    """Reproduce gravity_init() exactly, in code units. Returns g(r_code) callable."""
    G = G_CGS * UM * UT * UT / (UL**3)
    mc = np.log(1.0 + C_NFW) - C_NFW / (1.0 + C_NFW)
    H0_code = H0 * KM_S / MPC * UT
    rho_s = 200.0 * C_NFW**3 * H0_code**2 / (8.0 * np.pi * G * mc)
    M_nfw_c = M_NFW * MSUN / UM
    nfw_Rs = (M_nfw_c / (4.0 * np.pi * rho_s * mc)) ** (1.0 / 3.0)
    nfw_A = G * M_nfw_c / mc
    hq_R = R_BCG * KPC / UL
    hq_GM = G * (M_BCG * MSUN / UM)
    bh_GM = G * (M_BH * MSUN / UM)
    bh_eps2 = (EPS_SMBH * KPC / UL) ** 2

    def g_code(r):
        r2 = r * r
        x = r / nfw_Rs
        g = nfw_A / r2 * (np.log(1.0 + x) - x / (1.0 + x))
        rr = r + hq_R
        g += hq_GM / (rr * rr)
        s = r2 + bh_eps2
        g += bh_GM * r / (s * np.sqrt(s))
        return g

    return g_code, G


def K_erg(r_cm):
    """ACCEPT entropy profile K(r) in erg cm^2 (r in cm)."""
    return (K0 + K100 * (r_cm / (100.0 * KPC)) ** ALPHA_K) * KEV_ERG


def solve_hse(g_cgs, r_min_cm, r_max_cm, gamma):
    """Integrate dP/dr = -mu_e m_H n_e g, n_e = (P mu/(mu_e K))^(1/gamma), from r_ref out and in."""
    r_ref_cm = R_REF * KPC
    P_ref = (MU_E / MU) * K_erg(r_ref_cm) * N_E_REF ** gamma

    def dPdr(r, P):
        P = max(P[0], 1e-40)
        n_e = (P * MU / (MU_E * K_erg(r))) ** (1.0 / gamma)
        return [-MU_E * M_H * g_cgs(r) * n_e]

    sol_out = solve_ivp(dPdr, [r_ref_cm, r_max_cm], [P_ref], dense_output=True,
                        rtol=1e-8, atol=1e-30, max_step=(r_max_cm - r_ref_cm) / 2000)
    sol_in = solve_ivp(dPdr, [r_ref_cm, r_min_cm], [P_ref], dense_output=True,
                       rtol=1e-8, atol=1e-30, max_step=(r_ref_cm - r_min_cm) / 2000)

    rr = np.concatenate([np.linspace(r_min_cm, r_ref_cm, 4000)[:-1],
                         np.linspace(r_ref_cm, r_max_cm, 6000)])
    P = np.where(rr <= r_ref_cm, sol_in.sol(rr)[0], sol_out.sol(rr)[0])
    P = np.maximum(P, 1e-40)
    n_e = (P * MU / (MU_E * K_erg(rr))) ** (1.0 / gamma)
    rho = MU_E * M_H * n_e
    T = K_erg(rr) * n_e ** (gamma - 1.0) / K_B
    return rr, rho, P, n_e, T


def velocity_modes(n_modes, lam_min, lam_max, lam_peak, seed):
    """Draw the vector-Fourier mode set. Global, so every rank draws the same one.

    Each mode: A_vec * sin(k.r + phi) with A_vec perpendicular to k, so that
        v = curl(A_vec sin(k.r + phi)) = cos(k.r + phi) * (k x A_vec)
    is exactly div-free. Amplitude follows an inverse-parabolic spectrum on |k|, peaked at
    lam_peak, matching the 2D setup so both scripts sample the same energy-per-mode envelope.
    """
    rng = np.random.default_rng(seed)
    k_min = 2.0 * np.pi / lam_max
    k_max = 2.0 * np.pi / lam_min
    k_peak = 2.0 * np.pi / lam_peak
    modes = []
    for _ in range(n_modes):
        kmag = rng.uniform(k_min, k_max)
        # inverse-parabolic amplitude, peaked at k_peak, ->0 at band edges
        half = max(k_peak - k_min, k_max - k_peak)
        amp = max(0.0, 1.0 - ((kmag - k_peak) / half) ** 2)
        # random k direction, uniform on the unit sphere
        cos_t = 1.0 - 2.0 * rng.random()
        sin_t = np.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        phi_dir = rng.uniform(0, 2.0 * np.pi)
        khat = np.array([sin_t * np.cos(phi_dir), sin_t * np.sin(phi_dir), cos_t])
        k_vec = kmag * khat
        # random unit A perpendicular to k (project a Gaussian off khat, then normalise)
        a = rng.standard_normal(3)
        a -= np.dot(a, khat) * khat
        na = np.linalg.norm(a)
        if na < 1e-12:  # degenerate — try again with a fixed off-axis vector
            a = np.array([1.0, 0.0, 0.0]) - khat[0] * khat
            na = np.linalg.norm(a)
        a /= na
        # |k x A| = kmag (A perpendicular to k, |A|=1). We want |v| ~ amp, so scale by amp/kmag.
        psi_amp = amp / kmag
        modes.append((k_vec, np.cross(k_vec, a) * psi_amp, rng.uniform(0, 2.0 * np.pi)))
    return modes


def solenoidal_velocity_3d(pos, modes):
    """v = curl(A) at each cell. Unnormalised; the caller scales to sigma_v."""
    v = np.zeros_like(pos)
    for k_vec, cross, phase in modes:
        v += np.cos(pos @ k_vec + phase)[:, None] * cross[None, :]
    return v


# ============================================================
# Mesh
# ============================================================

def _layer_grid(n_side, lo, hi):
    """Cell-centre index range of a [lo, hi]^3 sub-box at spacing 1/n_side."""
    d = 1.0 / n_side
    i0 = int(lo / d)
    return i0, int(np.ceil(hi / d)) - i0, d


def _layers(args):
    """(n_side, lo, hi, r_in, r_out) per layer, finest first, plus the finest cell size."""
    refine = args.n_side_center > args.n and args.refine_radius_kpc > 0.0
    outer = args.n_side_outer > 0 and args.outer_transition_kpc > 0.0
    R_trans = args.outer_transition_kpc / args.L_box_kpc if outer else np.inf
    layers = []

    if refine:
        # factor-2 levels, each covering twice the radius of the finer one
        L = max(1, int(round(np.log2(args.n_side_center / args.n))))
        n_finest = args.n * (2 ** L)
        R_f = args.refine_radius_kpc / args.L_box_kpc
        for i in range(L):
            n_side = args.n * (2 ** (L - i))
            r_in = 0.0 if i == 0 else R_f * (2 ** (i - 1))
            r_out = R_f * (2 ** i)
            mgn = 4.0 / n_side
            layers.append((n_side, 0.5 - r_out - mgn, 0.5 + r_out + mgn, r_in, r_out))
        r_base_in = R_f * (2 ** (L - 1))
    else:
        n_finest = args.n
        r_base_in = 0.0

    if outer:
        mgn = 4.0 / args.n
        lo, hi = max(0.0, 0.5 - R_trans - mgn), min(1.0, 0.5 + R_trans + mgn)
        layers.append((args.n, lo, hi, r_base_in, R_trans))
        layers.append((args.n_side_outer, 0.0, 1.0, R_trans, np.inf))
    else:
        layers.append((args.n, 0.0, 1.0, r_base_in, np.inf))

    return layers, n_finest


def _layered_mesh(args):
    """Stacked jittered grids, each keeping only the cells in its radial band."""
    layers, n_finest = _layers(args)
    grids = [_layer_grid(n_side, lo, hi) for n_side, lo, hi, _, _ in layers]
    r_in = np.array([spec[3] for spec in layers])
    r_out = np.array([spec[4] for spec in layers])
    offsets = np.concatenate(([0], np.cumsum([n ** 3 for _, n, _ in grids])))

    def which(ids):
        li = np.searchsorted(offsets, ids, side="right") - 1
        return li, ids - offsets[li]

    def candidate(ids):
        li, k = which(ids)
        pos = np.empty((len(ids), 3), dtype=np.float64)
        dd = np.empty(len(ids), dtype=np.float64)
        for j, (i0, n, d) in enumerate(grids):
            m = li == j
            if not np.any(m):
                continue
            kk = k[m]
            # ravel order (y, x, z), matching np.meshgrid(indexing="xy")
            ix, iy, iz = (kk // n) % n, kk // (n * n), kk % n
            pos[m] = (np.column_stack((ix, iy, iz)).astype(np.float64) + i0 + 0.5) * d
            dd[m] = d
        for ax in range(3):
            pos[:, ax] += per_particle_signed(args.rng_seed, ids, axis=ax) * (args.perturbation * dd)
        return pos

    def accept(ids, pos):
        li, _ = which(ids)
        r = np.linalg.norm(pos - 0.5, axis=1)
        return (r >= r_in[li]) & (r < r_out[li])

    return Mesh(int(offsets[-1]), candidate, accept), n_finest


def _smooth_mesh(args):
    """One fine grid rejection-thinned to a smoothly varying cell size.

    Keep probability is set so the seed density matches a smoothstep from 1/n_side_center
    inside refine_radius to 1/n at r > transition_stop; accepted cells get extra jitter so
    the effective jitter stays proportional to the local cell size. No factor-2 tier
    interfaces, hence none of the volume-collapse mesh-reorganization artifacts.
    """
    n = args.n_side_center
    dx_center, dx_base = 1.0 / n, 1.0 / args.n
    R_plateau = args.refine_radius_kpc / args.L_box_kpc
    R_outer = args.transition_stop_kpc / args.L_box_kpc

    def base(ids):
        ix, iy, iz = (ids // n) % n, ids // (n * n), ids % n
        pos = (np.column_stack((ix, iy, iz)).astype(np.float64) + 0.5) * dx_center
        for ax in range(3):
            pos[:, ax] += per_particle_signed(args.rng_seed, ids, axis=ax) * (args.perturbation * dx_center)
        return pos

    def cell_size(pos):
        r = np.linalg.norm(pos - 0.5, axis=1)
        x = np.clip((r - R_plateau) / (R_outer - R_plateau), 0.0, 1.0)
        return dx_center + x * x * (3.0 - 2.0 * x) * (dx_base - dx_center)

    def candidate(ids):
        pos = base(ids)
        extra = args.perturbation * (cell_size(pos) - dx_center)
        for ax in range(3):
            pos[:, ax] += per_particle_signed(args.rng_seed, ids, axis=3 + ax) * extra
        return pos

    def accept(ids, _pos):
        # thin against the pre-extra-jitter radius, which is what sets the target density
        keep_prob = (dx_center / cell_size(base(ids))) ** 3
        return per_particle_uniform(args.rng_seed, ids, axis=6) < keep_prob

    return Mesh(n ** 3, candidate, accept), n


def build_cluster_mesh(args):
    """Mesh plus the finest designed cell size (the VOL_REGULARIZE reference)."""
    if args.smooth:
        if not (args.n_side_center > args.n and args.refine_radius_kpc > 0.0):
            raise SystemExit("--smooth requires --n_side_center > --n and --refine_radius_kpc > 0")
        if args.transition_stop_kpc <= args.refine_radius_kpc:
            raise SystemExit(f"--transition_stop_kpc ({args.transition_stop_kpc}) must exceed "
                             f"--refine_radius_kpc ({args.refine_radius_kpc})")
        return _smooth_mesh(args)
    return _layered_mesh(args)


PARAM_TEMPLATE = """# 3D cool-core cluster HSE testbed (gravity + cooling + AGN) — generated by create_cluster_3d.py
ic_file = {ic_file}
output_directory = ./output/

time_end = 5.0
output_dt = 0.01

CFL_frac = 0.3

rebalance_interval = 10
imbalance_log_interval = 1000
imbalance_threshold = 1.10

# mesh: finest designed cell size (code units) — VOL_REGULARIZE size reference (protects refinement)
vol_ref_cell_size = {vol_ref:.10g}

# code units (cgs per code unit)
UnitLength_in_cm = {UL:.8e}
UnitMass_in_g = {UM:.8e}
UnitVelocity_in_cm_per_s = {UV:.8e}

# gravity (physical units) — Fournier et al. Table 1
M_NFW = {M_NFW:.6e}
c_NFW = {C_NFW}
H0 = {H0}
M_BCG = {M_BCG:.6e}
R_BCG = {R_BCG}
M_BH = {M_BH:.6e}
smbh_softening = {EPS_SMBH}

# cooling (only read when COOLING is compiled in)
cooling_table = ics/cooling_table_schure2009.txt
T_floor = 1.0e4

# stellar feedback (only read when SF_FEEDBACK is compiled in) — Fournier et al. Sect. 2.5
Gamma_SNIa = 3.0e-14                      # SNIa rate [/yr/Msun]
E_SNIa = 1.0e51                          # energy per SNIa [erg]
alpha_SNIa = 1.0e-19                     # stellar mass-loss rate [/s]
eff_SF = 5.0e-5                          # particle-free SF heating efficiency
n_SF = 50.0                              # SF density threshold [n_H cm^-3]
T_SF = 2.0e4                             # SF temperature ceiling [K]
R_SF = 25.0                              # SF outer radius [kpc]

# AGN feedback (only read when AGN_THERMAL/AGN_KINETIC compiled in) — Sect. 2.4
# 3D geometry: paper values (0.5 kpc) are appropriate if the central resolution can resolve them;
# with the default uniform --n={n_side} (~{cell_kpc:.2f} kpc/cell) use --n_side_center
# + --refine_radius_kpc to nest a finer patch, or bump these radii to a few cell widths.
eta_agn = 0.01                           # accretion->energy efficiency
R_acc = 2.5                              # accretion radius [kpc] (paper 0.5; bumped to span ~few finest cells)
T_cold_acc = 5.0e4                       # cold-accretion temperature threshold [K]
t_acc = 5.0                              # accretion timescale [Myr]
R_T = 2.5                                # thermal deposition radius [kpc] (paper 0.5; bumped to span ~few finest cells)
f_T = 0.25                               # thermal feedback fraction
f_K = 0.75                               # kinetic feedback fraction
R_jet = 5.0                              # jet cross-radius [kpc]
h_jet = 3.0                              # jet launch-zone thickness [kpc]
L_jet = 5.0                              # jet launch-zone offset from center [kpc]
v_cap = 0.05                             # velocity cap [fraction of c]

# central-region hard caps (LIMITERS) — Fournier et al. Sect. 2.1
R_lim = 20.0                             # radius of clamped region [kpc]
T_max_lim = 5.0e9                        # temperature ceiling in r<R_lim [K]
v_cap_lim = 0.05                         # |v| ceiling in r<R_lim [fraction of c]
"""


def main():
    parser = build_arg_parser(
        "cluster",
        default_n=128,              # base resolution per axis
        default_perturbation=0.25,  # grid jitter as a fraction of the local cell size
        default_rng_seed=20260709,
        # the layered mesh replaces mesh_mode, and the box is [0,1] code units by
        # construction -- physical size is set by --L_box_kpc
        fixed={"dimension": 3, "mesh_mode": "cartesian", "extent": 1.0},
    )
    parser.add_argument("--param", default="param_cluster_3d.txt")
    # emulated static refinement: a denser central cube (n_side_center) inside refine_radius_kpc.
    # 0 (default) -> uniform. Cheap way to get high central resolution where the cool core / AGN lives.
    parser.add_argument("--n_side_center", type=int, default=0)
    parser.add_argument("--refine_radius_kpc", type=float, default=0.0)
    parser.add_argument("--L_box_kpc", type=float, default=800.0)  # paper (Fournier et al.) uses 800 kpc
    # coarse outer shell: fills r > outer_transition_kpc at n_side_outer resolution across the box,
    # so we can grow the box to the paper's 800 kpc without paying for base resolution in the tenuous
    # outskirts. Set n_side_outer=0 to disable (base fills the whole box).
    parser.add_argument("--n_side_outer", type=int, default=100)
    parser.add_argument("--outer_transition_kpc", type=float, default=150.0)
    parser.add_argument("--smooth", action="store_true",
                        help="use smooth-varying resolution instead of nested factor-2 tiers")
    parser.add_argument("--transition_stop_kpc", type=float, default=250.0,
                        help="radius at which cell size reaches L_box/n (smooth mode only)")
    parser.add_argument("--unit_velocity_cgs", type=float, default=1.0e8)  # 1000 km/s
    parser.add_argument("--sigma_v_kms", type=float, default=75.0)
    parser.add_argument("--n_modes", type=int, default=40)
    args = parser.parse_args()

    comm, rank, _nranks = mpi_runtime()
    root = rank == 0

    # unit system: box = L_box; density unit chosen so rho_code ~ 1 at the reference density
    UL = args.L_box_kpc * KPC
    UV = args.unit_velocity_cgs
    UT = UL / UV
    U_density = MU_E * M_H * N_E_REF          # rho_code = 1 at n_e = n_e_ref
    UM = U_density * UL**3
    U_pressure = U_density * UV * UV

    g_code, _G = build_gravity_code_units(UL, UM, UT)
    g_cgs = lambda r_cm: g_code(r_cm / UL) * UV * UV / UL   # convert code accel -> cgs

    # HSE profile out to the box corner (sqrt(3)/2 * L for a cube). Deterministic, so every
    # rank solves it and they agree without communicating.
    rr, rho_cgs, P_cgs, n_e, T = solve_hse(g_cgs, 0.05 * KPC, 0.88 * UL, args.gamma)
    prof_rho = interp1d(rr, rho_cgs, bounds_error=False, fill_value=(rho_cgs[0], rho_cgs[-1]))
    prof_P = interp1d(rr, P_cgs, bounds_error=False, fill_value=(P_cgs[0], P_cgs[-1]))

    if root:
        print(f"HSE profile: n_e(10kpc)={np.interp(10*KPC, rr, n_e):.4g} cm^-3 (target {N_E_REF})")
        for rk in [5, 10, 30, 50, 100]:
            print(f"  r={rk:4d} kpc: n_e={np.interp(rk*KPC, rr, n_e):.3g}  T={np.interp(rk*KPC, rr, T):.3g} K")

    mesh, n_finest = build_cluster_mesh(args)
    modes = velocity_modes(args.n_modes, 12.5 / args.L_box_kpc, 50.0 / args.L_box_kpc,
                           25.0 / args.L_box_kpc, args.rng_seed)
    sigma_code = args.sigma_v_kms * KM_S / UV

    if root:
        print(f"mesh: {mesh.n_global} cells, finest {args.L_box_kpc/n_finest:.2f} kpc/cell, "
              f"base {args.L_box_kpc/args.n:.2f} kpc/cell")

    def fill(row_lo, n_local, _args):
        # wrap after the layer cut, which is made on the unwrapped radius
        pos = mesh.positions(row_lo, n_local) % 1.0

        r_cm = np.linalg.norm(pos - 0.5, axis=1) * UL
        rho = prof_rho(r_cm) / U_density
        P = prof_P(r_cm) / U_pressure

        vel = solenoidal_velocity_3d(pos, modes)
        sq, cnt = float(np.sum(vel * vel)), len(vel)
        if comm is not None:
            from mpi4py import MPI
            sq = comm.allreduce(sq, op=MPI.SUM)
            cnt = comm.allreduce(cnt, op=MPI.SUM)
        vel *= sigma_code / np.sqrt(sq / cnt)

        energy = P / (args.gamma - 1.0) + 0.5 * rho * np.sum(vel * vel, axis=1)
        return pos, vel, rho, energy

    filename = resolve_filename(args, "cluster")
    write_ic(filename, mesh.n_global, args.dimension, fill, args)

    if root:
        with open(args.param, "w") as f:
            f.write(PARAM_TEMPLATE.format(
                ic_file=filename, vol_ref=1.0 / n_finest, UL=UL, UM=UM, UV=UV,
                M_NFW=M_NFW, C_NFW=C_NFW, H0=H0, M_BCG=M_BCG, R_BCG=R_BCG,
                M_BH=M_BH, EPS_SMBH=EPS_SMBH, n_side=args.n,
                cell_kpc=args.L_box_kpc / args.n,
            ))
        print(f"Wrote {args.param}")


if __name__ == "__main__":
    main()
