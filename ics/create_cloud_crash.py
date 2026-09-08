"""
3D cloud-crash initial conditions.

Spherical clouds placed on a sphere around the box centre and aimed at the center with small offset.
Spheres have 1-tanh density profiles and small surface perturbation. Pressure is uniform.

There is no physical motivation behind this ic. It just looks interesting :D

Runs in either mode:
  python create_cloud_crash.py --n 200
  mpirun -np 4 python create_cloud_crash.py --n 400
"""

import numpy as np

from common import (
    build_arg_parser,
    per_particle_normal,
    resolve_filename,
    seed_positions_slice,
    write_ic,
)

CLOUD_START_RADIUS = 0.32        # how far clouds start from centre (fraction of extent)
CLOUD_RADIUS_RANGE = (0.06, 0.10)
CLOUD_DENSITY_CONTRAST = 10.0    # cloud rho / ambient rho
AMBIENT_RHO = 0.1
AMBIENT_P = 0.2
CLOUD_SPEED = 1.5                # |v| of each cloud (subsonic in cloud, transonic in ambient)
AIM_JITTER = 0.18                # rms perpendicular jitter in aim direction (rad-ish)
EDGE_SOFTNESS = 0.18             # tanh transition width as fraction of cloud radius
SURFACE_PERTURB = 0.10           # surface deformation amplitude (frac of radius)
CENTRAL_TURBULENCE = 0.04        # rms velocity perturbation amplitude near centre


def fibonacci_sphere(n):
    """Return n unit vectors approximately uniformly distributed on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    phi = golden * np.arange(n)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def cloud_field(args):
    """Cloud centres, radii and velocities. Only n_clouds draws, so every rank gets the same set."""
    rng = np.random.default_rng(args.rng_seed)

    # even coverage of the launch sphere, then a small random rotation
    dirs = fibonacci_sphere(args.n_clouds)
    rot_axis = rng.normal(size=3)
    rot_axis /= np.linalg.norm(rot_axis)
    rot_ang = rng.uniform(0, 2 * np.pi)
    K = np.array([[0, -rot_axis[2], rot_axis[1]],
                  [rot_axis[2], 0, -rot_axis[0]],
                  [-rot_axis[1], rot_axis[0], 0]])
    R = np.eye(3) + np.sin(rot_ang) * K + (1 - np.cos(rot_ang)) * K @ K
    dirs = dirs @ R.T

    centres = 0.5 * args.extent + CLOUD_START_RADIUS * args.extent * dirs
    radii = rng.uniform(*CLOUD_RADIUS_RANGE, size=args.n_clouds)

    # aim at the centre with random perpendicular jitter
    vels = np.zeros((args.n_clouds, 3), dtype=np.float64)
    for i in range(args.n_clouds):
        aim = -dirs[i]
        tmp = np.array([1.0, 0.0, 0.0]) if abs(aim[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(aim, tmp)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(aim, e1)
        jx, jy = rng.normal(0, AIM_JITTER, 2)
        v_dir = aim + jx * e1 + jy * e2
        v_dir /= np.linalg.norm(v_dir)
        vels[i] = CLOUD_SPEED * v_dir

    return centres, radii, vels


def fill_cloud_crash(row_lo, n_local, args):
    """Compute (pos, vel, rho, energy) for global rows [row_lo, row_lo + n_local)."""
    pos = seed_positions_slice(
        row_lo, n_local, args.n ** args.dimension,
        dimension=args.dimension,
        extent=args.extent,
        rng_seed=args.rng_seed,
        mesh_mode=args.mesh_mode,
        perturbation=args.perturbation,
    )
    centres, radii, vels = cloud_field(args)

    rho = np.full(n_local, AMBIENT_RHO, dtype=np.float64)
    v_accum = np.zeros((n_local, 3), dtype=np.float64)
    w_accum = np.full(n_local, 1e-3, dtype=np.float64)

    for i in range(args.n_clouds):
        rel = pos - centres[i]
        d = np.linalg.norm(rel, axis=1) + 1e-30

        # Y_42-like surface deformation: sin^2(theta) cos(2 phi) modulated, roughly in [-1, 1]
        cos_th = rel[:, 2] / d
        phi_a = np.arctan2(rel[:, 1], rel[:, 0])
        y42 = (1.0 - cos_th * cos_th) * np.cos(2.0 * phi_a) * (7.0 * cos_th * cos_th - 1.0) * 0.5
        eff_r = radii[i] * (1.0 + SURFACE_PERTURB * y42)

        f = 0.5 * (1.0 - np.tanh((d - eff_r) / (EDGE_SOFTNESS * radii[i])))  # 1 inside, 0 outside

        rho += (CLOUD_DENSITY_CONTRAST - 1.0) * AMBIENT_RHO * f
        v_accum += vels[i] * f[:, None]
        w_accum += f

    vel = v_accum / w_accum[:, None]

    # small turbulent kick concentrated near the centre
    r_c = np.linalg.norm(pos - 0.5 * args.extent, axis=1)
    envelope = CENTRAL_TURBULENCE * np.exp(-(r_c / (0.18 * args.extent)) ** 2)
    ids = np.arange(row_lo, row_lo + n_local, dtype=np.int64)
    for ax in range(3):
        vel[:, ax] += per_particle_normal(args.rng_seed, ids, axis=ax) * envelope

    energy = AMBIENT_P / (args.gamma - 1.0) + 0.5 * rho * np.sum(vel ** 2, axis=1)
    return pos, vel, rho, energy


if __name__ == "__main__":
    parser = build_arg_parser(
        "cloud_crash",
        default_n=200,
        default_mesh_mode="cartesian",
        default_rng_seed=20250429,
        fixed={"dimension": 3},
    )
    parser.add_argument("--n_clouds", type=int, default=8)
    args = parser.parse_args()

    write_ic(
        filename=resolve_filename(args, "cloud_crash"),
        n_global=args.n ** args.dimension,
        dimension=args.dimension,
        fill_fn=fill_cloud_crash,
        args=args,
    )
