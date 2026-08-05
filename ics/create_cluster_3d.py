"""
3D idealised cool-core cluster in hydrostatic equilibrium.

Native geometry for the setup of Fournier et al. (magnetised cold filaments, their Sect. 2.3): NFW
+ Hernquist(BCG) + softened SMBH point mass (their Eqs. 3-5) closed by the ACCEPT entropy profile
K(r)=K0+K100*(r/100kpc)^alpha_K (Eqs. 6-7) at n_e_ref@r_ref. Same code-unit constants that
src/astro/gravity.cu::gravity_init builds, so the IC is in HSE against gravity_accel.

Symmetry-breaking velocity field is solenoidal in 3D (v = curl(A), 40 vector-Fourier modes,
lambda in [12.5, 50] kpc, inverse-parabolic power spectrum peaked at 25 kpc, sigma_v).

Writes the IC hdf5 AND a matching param file so the unit system can never drift between them.
"""

import argparse
import numpy as np
import h5py
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# cgs constants
G_CGS = 6.67430e-8
KPC = 3.085677581e21
MPC = 3.085677581e24
MSUN = 1.989e33
M_H = 1.67262192e-24
K_B = 1.380649e-16
KEV_ERG = 1.602176634e-9
KM_S = 1.0e5
GAMMA = 5.0 / 3.0
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


def solve_hse(g_cgs, r_min_cm, r_max_cm):
    """Integrate dP/dr = -mu_e m_H n_e g, with n_e = (P mu/(mu_e K))^(3/5), from r_ref outward+inward."""
    r_ref_cm = R_REF * KPC
    P_ref = (MU_E / MU) * K_erg(r_ref_cm) * N_E_REF ** (5.0 / 3.0)

    def dPdr(r, P):
        P = max(P[0], 1e-40)
        n_e = (P * MU / (MU_E * K_erg(r))) ** (3.0 / 5.0)
        return [-MU_E * M_H * g_cgs(r) * n_e]

    sol_out = solve_ivp(dPdr, [r_ref_cm, r_max_cm], [P_ref], dense_output=True,
                        rtol=1e-8, atol=1e-30, max_step=(r_max_cm - r_ref_cm) / 2000)
    sol_in = solve_ivp(dPdr, [r_ref_cm, r_min_cm], [P_ref], dense_output=True,
                       rtol=1e-8, atol=1e-30, max_step=(r_ref_cm - r_min_cm) / 2000)

    rr = np.concatenate([np.linspace(r_min_cm, r_ref_cm, 4000)[:-1],
                         np.linspace(r_ref_cm, r_max_cm, 6000)])
    P = np.where(rr <= r_ref_cm, sol_in.sol(rr)[0], sol_out.sol(rr)[0])
    P = np.maximum(P, 1e-40)
    n_e = (P * MU / (MU_E * K_erg(rr))) ** (3.0 / 5.0)
    rho = MU_E * M_H * n_e
    T = K_erg(rr) * n_e ** (2.0 / 3.0) / K_B
    return rr, rho, P, n_e, T


def solenoidal_velocity_3d(pos, n_modes, lam_min, lam_max, lam_peak, seed):
    """3D divergence-free velocity from a random vector-potential; v = curl(A). RMS normalised later.

    Each mode: A_vec * sin(k.r + phi) with A_vec chosen perpendicular to k so that
        v = curl(A_vec sin(k.r + phi)) = cos(k.r + phi) * (k x A_vec)
    is exactly div-free. Amplitude follows an inverse-parabolic spectrum on |k|, peaked at lam_peak,
    matching the 2D setup so that both scripts sample the same energy-per-mode envelope.
    """
    rng = np.random.default_rng(seed)
    v = np.zeros_like(pos)  # (N, 3)
    k_min = 2.0 * np.pi / lam_max
    k_max = 2.0 * np.pi / lam_min
    k_peak = 2.0 * np.pi / lam_peak
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
        # This matches the 2D convention where psi_amp = amp / kmag.
        psi_amp = amp / kmag
        cross = np.cross(k_vec, a) * psi_amp     # (3,)
        phi_wave = rng.uniform(0, 2.0 * np.pi)
        arg = pos @ k_vec + phi_wave              # (N,)
        v += np.cos(arg)[:, None] * cross[None, :]
    return v[:, 0], v[:, 1], v[:, 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--filename", default="ics/IC_cluster_3D.hdf5")
    p.add_argument("--param", default="ics/param_cluster_3d.txt")
    p.add_argument("--n_side", type=int, default=128)  # base resolution (per axis)
    # emulated static refinement: a denser central cube (n_side_center) inside refine_radius_kpc.
    # 0 (default) -> uniform. Cheap way to get high central resolution where the cool core / AGN lives.
    p.add_argument("--n_side_center", type=int, default=0)
    p.add_argument("--refine_radius_kpc", type=float, default=0.0)
    p.add_argument("--L_box_kpc", type=float, default=800.0)  # paper (Fournier et al.) uses 800 kpc
    # coarse outer shell: fills r > outer_transition_kpc at n_side_outer resolution across the box,
    # so we can grow the box to the paper's 800 kpc without paying for base resolution in the tenuous
    # outskirts. Set n_side_outer=0 to disable (base fills the whole box).
    p.add_argument("--n_side_outer", type=int, default=100)
    p.add_argument("--outer_transition_kpc", type=float, default=150.0)
    # Smooth-varying resolution mode (alternative to nested factor-2 tiers).
    # Cell size s(r) is a smoothstep from L_box/n_side_center at r<refine_radius_kpc
    # to L_box/n_side at r>transition_stop_kpc. Eliminates the discrete tier interfaces
    # that produce the volume-collapse mesh-reorganization artifacts.
    p.add_argument("--smooth", action="store_true",
                   help="Use smooth-varying resolution instead of nested factor-2 tiers")
    p.add_argument("--transition_stop_kpc", type=float, default=250.0,
                   help="Radius at which cell size reaches L_box/n_side (smooth mode only)")
    p.add_argument("--unit_velocity_cgs", type=float, default=1.0e8)  # 1000 km/s
    p.add_argument("--sigma_v_kms", type=float, default=75.0)
    p.add_argument("--n_modes", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260709)
    args = p.parse_args()

    # unit system: box = L_box; density unit chosen so rho_code ~ 1 at the reference density
    UL = args.L_box_kpc * KPC
    UV = args.unit_velocity_cgs
    UT = UL / UV
    U_density = MU_E * M_H * N_E_REF          # rho_code = 1 at n_e = n_e_ref
    UM = U_density * UL**3
    U_pressure = U_density * UV * UV

    g_code, G = build_gravity_code_units(UL, UM, UT)
    g_cgs = lambda r_cm: g_code(r_cm / UL) * UV * UV / UL   # convert code accel -> cgs

    # HSE profile out to the box corner (sqrt(3)/2 * L for a cube)
    r_min_cm = 0.05 * KPC
    r_max_cm = 0.88 * UL
    rr, rho_cgs, P_cgs, n_e, T = solve_hse(g_cgs, r_min_cm, r_max_cm)
    prof_rho = interp1d(rr, rho_cgs, bounds_error=False, fill_value=(rho_cgs[0], rho_cgs[-1]))
    prof_P = interp1d(rr, P_cgs, bounds_error=False, fill_value=(P_cgs[0], P_cgs[-1]))

    print(f"HSE profile: n_e(10kpc)={np.interp(10*KPC, rr, n_e):.4g} cm^-3 (target {N_E_REF})")
    for rk in [5, 10, 30, 50, 100]:
        print(f"  r={rk:4d} kpc: n_e={np.interp(rk*KPC, rr, n_e):.3g}  T={np.interp(rk*KPC, rr, T):.3g} K")

    # jittered 3D grid (optionally two-level nested: coarse box + refined central sphere)
    rng = np.random.default_rng(args.seed)

    def jittered_grid(n_side, x0=0.0, x1=1.0, y0=0.0, y1=1.0, z0=0.0, z1=1.0):
        d = 1.0 / n_side
        cx = np.arange(int(x0 / d), int(np.ceil(x1 / d))) * d + 0.5 * d
        cy = np.arange(int(y0 / d), int(np.ceil(y1 / d))) * d + 0.5 * d
        cz = np.arange(int(z0 / d), int(np.ceil(z1 / d))) * d + 0.5 * d
        xx, yy, zz = np.meshgrid(cx, cy, cz, indexing="xy")
        p = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float64)
        return p + rng.uniform(-0.25 * d, 0.25 * d, size=p.shape), d

    # multi-level nested refinement: factor-2 levels, each covering twice the radius of the finer one.
    # refine_radius_kpc is the FINEST level's radius; base resolution fills to outer_transition_kpc
    # (or the whole box if the outer shell is disabled), then n_side_outer fills the rest coarsely.
    refine = args.n_side_center > args.n_side and args.refine_radius_kpc > 0.0
    outer = args.n_side_outer > 0 and args.outer_transition_kpc > 0.0
    R_trans = args.outer_transition_kpc / args.L_box_kpc if outer else float("inf")
    parts, desc = [], []

    # Smooth-varying resolution mode: rejection-subsample a fine jittered grid at n_side_center,
    # with local keep probability set so the resulting seed density matches a smoothstep from
    # s_min = 1/n_side_center inside R_plateau to s_max = 1/n_side at r>R_outer. Each accepted seed
    # then gets extra jitter proportional to (s_local - s_min) so the effective jitter is ±0.25*s_local
    # everywhere. No factor-2 tier interfaces → no volume-collapse mesh-reorganization artifact.
    if args.smooth:
        if not refine:
            raise SystemExit("--smooth requires --n_side_center > --n_side and --refine_radius_kpc > 0")

        dx_center = 1.0 / args.n_side_center
        dx_base   = 1.0 / args.n_side
        R_plateau = args.refine_radius_kpc / args.L_box_kpc
        R_outer   = args.transition_stop_kpc / args.L_box_kpc
        if R_outer <= R_plateau:
            raise SystemExit(f"--transition_stop_kpc ({args.transition_stop_kpc}) must exceed --refine_radius_kpc ({args.refine_radius_kpc})")

        print(f"Smooth mode: fine grid {args.n_side_center}^3 ({args.L_box_kpc/args.n_side_center:.2f} kpc), "
              f"plateau r<{args.refine_radius_kpc:.0f} kpc, smoothstep to {args.L_box_kpc/args.n_side:.2f} kpc at r>{args.transition_stop_kpc:.0f} kpc")

        p_fine, _ = jittered_grid(args.n_side_center)
        r_fine = np.sqrt((p_fine[:, 0] - 0.5) ** 2 + (p_fine[:, 1] - 0.5) ** 2 + (p_fine[:, 2] - 0.5) ** 2)
        x = np.clip((r_fine - R_plateau) / (R_outer - R_plateau), 0.0, 1.0)
        s_local = dx_center + x * x * (3.0 - 2.0 * x) * (dx_base - dx_center)  # smoothstep
        keep_prob = (dx_center / s_local) ** 3
        u = rng.uniform(0.0, 1.0, size=p_fine.shape[0])
        keep = u < keep_prob

        accepted = p_fine[keep].copy()
        s_acc = s_local[keep]
        # extra jitter: fine grid already carries ±0.25*dx_center, add ±0.25*(s_local - dx_center)
        # so combined effective jitter is ±0.25*s_local at every accepted cell
        extra_amp = 0.25 * (s_acc - dx_center)
        accepted += rng.uniform(-1.0, 1.0, size=accepted.shape) * extra_amp[:, None]
        accepted = np.mod(accepted, 1.0)

        parts.append(accepted)
        desc.append(f"smooth: {args.n_side_center}^3 fine ({args.L_box_kpc/args.n_side_center:.2f}kpc) "
                    f"plateau r<{args.refine_radius_kpc:.0f}kpc smoothstep to "
                    f"{args.L_box_kpc/args.n_side:.2f}kpc at r>{args.transition_stop_kpc:.0f}kpc")
        n_finest = args.n_side_center
    elif refine:
        L = max(1, int(round(np.log2(args.n_side_center / args.n_side))))  # number of refinement steps
        n_finest = args.n_side * (2 ** L)                                  # actual finest (power-of-2 multiple)
        R_f = args.refine_radius_kpc / args.L_box_kpc                      # finest-level radius (code)
        for i in range(L):  # i=0 finest, radius doubles as we coarsen
            n_side = args.n_side * (2 ** (L - i))
            r_in = 0.0 if i == 0 else R_f * (2 ** (i - 1))
            r_out = R_f * (2 ** i)
            mgn = 4.0 / n_side
            lo, hi = 0.5 - r_out - mgn, 0.5 + r_out + mgn
            p, _ = jittered_grid(n_side, lo, hi, lo, hi, lo, hi)
            r = np.sqrt((p[:, 0] - 0.5) ** 2 + (p[:, 1] - 0.5) ** 2 + (p[:, 2] - 0.5) ** 2)
            parts.append(p[(r >= r_in) & (r < r_out)])
            desc.append(f"{n_side}^3 ({args.L_box_kpc/n_side:.2f}kpc) r<{r_out*args.L_box_kpc:.0f}")
        r_base_in = R_f * (2 ** (L - 1))
    else:
        n_finest = args.n_side
        r_base_in = 0.0

    if not args.smooth:
        # base layer: r_base_in <= r < r_base_out (either R_trans or box corner if outer disabled)
        if outer:
            mgn = 4.0 / args.n_side
            lo_b, hi_b = max(0.0, 0.5 - R_trans - mgn), min(1.0, 0.5 + R_trans + mgn)
            pb, _ = jittered_grid(args.n_side, lo_b, hi_b, lo_b, hi_b, lo_b, hi_b)
        else:
            pb, _ = jittered_grid(args.n_side)
        rb = np.sqrt((pb[:, 0] - 0.5) ** 2 + (pb[:, 1] - 0.5) ** 2 + (pb[:, 2] - 0.5) ** 2)
        if outer:
            parts.append(pb[(rb >= r_base_in) & (rb < R_trans)])
            desc.append(f"{args.n_side}^3 ({args.L_box_kpc/args.n_side:.2f}kpc) {r_base_in*args.L_box_kpc:.0f}<r<{args.outer_transition_kpc:.0f}")
        else:
            parts.append(pb[rb >= r_base_in])
            desc.append(f"{args.n_side}^3 ({args.L_box_kpc/args.n_side:.2f}kpc) r>{r_base_in*args.L_box_kpc:.0f}")

        # outer coarse shell: r > R_trans out to box corners
        if outer:
            po, _ = jittered_grid(args.n_side_outer)
            ro = np.sqrt((po[:, 0] - 0.5) ** 2 + (po[:, 1] - 0.5) ** 2 + (po[:, 2] - 0.5) ** 2)
            parts.append(po[ro >= R_trans])
            desc.append(f"{args.n_side_outer}^3 ({args.L_box_kpc/args.n_side_outer:.2f}kpc) r>{args.outer_transition_kpc:.0f}")

    pos = np.vstack(parts)
    print(f"{len(desc)}-level grid: " + "  |  ".join(reversed(desc)))
    pos %= 1.0
    n = len(pos)

    r_code = np.sqrt((pos[:, 0] - 0.5) ** 2 + (pos[:, 1] - 0.5) ** 2 + (pos[:, 2] - 0.5) ** 2)
    r_cm = r_code * UL
    rho = prof_rho(r_cm) / U_density
    P = prof_P(r_cm) / U_pressure

    # solenoidal velocity perturbation, normalised to sigma_v
    vx, vy, vz = solenoidal_velocity_3d(pos, args.n_modes, 12.5 / args.L_box_kpc,
                                        50.0 / args.L_box_kpc, 25.0 / args.L_box_kpc, args.seed)
    rms = np.sqrt(np.mean(vx**2 + vy**2 + vz**2))
    sigma_code = args.sigma_v_kms * KM_S / UV
    vx *= sigma_code / rms
    vy *= sigma_code / rms
    vz *= sigma_code / rms
    vel = np.column_stack((vx, vy, vz))

    energy = P / (GAMMA - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)

    print(f"units: UnitLength={UL:.6e}  UnitMass={UM:.6e}  UnitVelocity={UV:.6e}")
    print(f"rho_code range [{rho.min():.3g}, {rho.max():.3g}], "
          f"v_perturb rms={sigma_code:.4g} code ({args.sigma_v_kms} km/s), c_s(core)~"
          f"{np.sqrt(GAMMA*P.max()/rho.max()):.3g} code")

    with h5py.File(args.filename, "w") as f:
        f.create_group("header").attrs["dimension"] = 3
        f.create_group("mesh").create_dataset("pos", data=pos)
        hyd = f.create_group("hydro")
        hyd.create_dataset("rho", data=rho)
        hyd.create_dataset("vel", data=vel)
        hyd.create_dataset("energy", data=energy)
    print(f"Wrote {args.filename} ({n} cells, {args.L_box_kpc} kpc box, {args.L_box_kpc/args.n_side:.2f} kpc/cell base)")

    # finest designed cell size (code units, box=[0,1]) — VOL_REGULARIZE reference so a refined
    # IC isn't de-refined by the size-equalizing drift. Equals 1/n_side for a uniform IC.
    vol_ref = 1.0 / n_finest

    with open(args.param, "w") as f:
        f.write(f"""# 3D cool-core cluster HSE testbed (gravity + cooling + AGN) — generated by create_cluster_3d.py
ic_file = {args.filename}
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
# with the default uniform n_side={args.n_side} (~{args.L_box_kpc/args.n_side:.2f} kpc/cell) use --n_side_center
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
""")
    print(f"Wrote {args.param}")


if __name__ == "__main__":
    main()
