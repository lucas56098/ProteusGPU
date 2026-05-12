"""
3D cloud-crash initial conditions.

Spherical clouds placed on a sphere around the box centre and aimed at the center with small offset.
Spheres have 1-tanh density profiles and small surface perturbation. Pressure is uniform.

There is no physical motivation behind this ic. It just looks interesting :D
"""

import h5py
import numpy as np
from common import seed_positions, build_arg_parser, resolve_filename


def fibonacci_sphere(n):
    """Return n unit vectors approximately uniformly distributed on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    phi = golden * np.arange(n)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def create_cloud_crash(
    filename,
    num_seeds,
    extent=1.0,
    gamma=5.0 / 3.0,
    n_clouds=8,
    cloud_start_radius=0.32,        # how far clouds start from centre (fraction of extent)
    cloud_radius_range=(0.06, 0.10),
    cloud_density_contrast=10.0,    # cloud rho / ambient rho
    ambient_rho=0.1,
    ambient_p=0.2,
    cloud_speed=1.5,                # |v| of each cloud (subsonic in cloud, transonic in ambient)
    aim_jitter=0.18,                # rms perpendicular jitter in aim direction (rad-ish)
    edge_softness=0.18,             # tanh transition width as fraction of cloud radius
    surface_perturb=0.10,           # surface deformation amplitude (frac of radius)
    central_turbulence=0.04,        # rms velocity perturbation amplitude near centre
    mesh_mode="random",             # ["random", "cartesian"]
    rng_seed=20250429,
):
    dimension = 3
    rng = np.random.default_rng(rng_seed)

    print(f"Creating cloud-crash IC: {filename}")
    print(f"  Total seeds:      {num_seeds}")
    print(f"  Mesh mode:        {mesh_mode}")
    print(f"  Number of clouds: {n_clouds}")
    print(f"  Cloud speed:      {cloud_speed}")
    print(f"  Density contrast: {cloud_density_contrast}")

    pos = seed_positions(num_seeds, dimension, extent=extent, mesh_mode=mesh_mode)

    centre = np.array([0.5 * extent] * 3)

    # Cloud placement: even coverage of the launch sphere
    cloud_dirs = fibonacci_sphere(n_clouds)

    # small random rotation
    rot_axis = rng.normal(size=3); rot_axis /= np.linalg.norm(rot_axis)
    rot_ang = rng.uniform(0, 2 * np.pi)
    K = np.array([[0, -rot_axis[2], rot_axis[1]],
                  [rot_axis[2], 0, -rot_axis[0]],
                  [-rot_axis[1], rot_axis[0], 0]])
    R = np.eye(3) + np.sin(rot_ang) * K + (1 - np.cos(rot_ang)) * K @ K
    cloud_dirs = cloud_dirs @ R.T

    cloud_centres = centre + cloud_start_radius * extent * cloud_dirs
    cloud_radii = rng.uniform(*cloud_radius_range, size=n_clouds)

    # Velocity: aim at centre with random perpendicular jitter
    cloud_vels = np.zeros((n_clouds, 3))
    for i in range(n_clouds):
        aim = -cloud_dirs[i]

        # build orthonormal frame around aim
        tmp = np.array([1.0, 0.0, 0.0]) if abs(aim[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(aim, tmp); e1 /= np.linalg.norm(e1)
        e2 = np.cross(aim, e1)
        
        # perpendicular jitter
        jx, jy = rng.normal(0, aim_jitter, 2)
        v_dir = aim + jx * e1 + jy * e2
        v_dir /= np.linalg.norm(v_dir)
        cloud_vels[i] = cloud_speed * v_dir

    # fields
    rho = np.full(num_seeds, ambient_rho)
    v_accum = np.zeros((num_seeds, 3))
    w_accum = np.full(num_seeds, 1e-3)

    for i in range(n_clouds):
        rel = pos - cloud_centres[i]
        d = np.linalg.norm(rel, axis=1) + 1e-30

        # spherical-harmonic-style surface deformation
        cos_th = rel[:, 2] / d
        phi_a = np.arctan2(rel[:, 1], rel[:, 0])
        
        # Y_42-like: sin^2(theta) cos(2 phi) modulated; values in roughly [-1, 1]
        y42 = (1.0 - cos_th * cos_th) * np.cos(2.0 * phi_a) * (7.0 * cos_th * cos_th - 1.0) * 0.5
        eff_r = cloud_radii[i] * (1.0 + surface_perturb * y42)

        edge_w = edge_softness * cloud_radii[i]
        f = 0.5 * (1.0 - np.tanh((d - eff_r) / edge_w))  # 1 inside, 0 outside

        rho += (cloud_density_contrast - 1.0) * ambient_rho * f
        v_accum += cloud_vels[i] * f[:, None]
        w_accum += f

    vel = v_accum / w_accum[:, None]

    # small turbulent kick concentrated near the centre
    r_c = np.linalg.norm(pos - centre, axis=1)
    central_envelope = np.exp(-(r_c / (0.18 * extent)) ** 2)
    vel += rng.normal(0.0, central_turbulence, vel.shape) * central_envelope[:, None]

    # uniform pressure
    pressure = np.full(num_seeds, ambient_p)

    energy = pressure / (gamma - 1.0) + 0.5 * rho * np.sum(vel ** 2, axis=1)

    cs_amb = np.sqrt(gamma * ambient_p / ambient_rho)
    cs_cloud = np.sqrt(gamma * ambient_p / (cloud_density_contrast * ambient_rho))
    print("\n  Initial state summary:")
    print(f"    rho range:          [{rho.min():.4f}, {rho.max():.4f}]")
    print(f"    pressure (uniform): {pressure[0]:.4f}")
    print(f"    |v| range:          [{np.linalg.norm(vel,axis=1).min():.4f}, "
          f"{np.linalg.norm(vel,axis=1).max():.4f}]")
    print(f"    c_s (ambient):      {cs_amb:.3f}")
    print(f"    c_s (cloud):        {cs_cloud:.3f}")
    print(f"    Mach (cloud→amb):   {cloud_speed / cs_amb:.2f}")
    print(f"    Mach (cloud-int):   {cloud_speed / cs_cloud:.2f}")
    print(f"    energy range:       [{energy.min():.4f}, {energy.max():.4f}]")

    with h5py.File(filename, "w") as f:
        header_group = f.create_group("header")
        header_group.attrs["dimension"] = dimension
        header_group.attrs["extent"] = extent
        header_group.attrs["gamma"] = gamma

        f.create_dataset("pos", data=pos)
        f.create_dataset("rho", data=rho)
        f.create_dataset("vel", data=vel)
        f.create_dataset("energy", data=energy)

    print(f"\nSuccessfully created {filename}\n")


if __name__ == "__main__":
    # cloud_crash is 3D-only.
    parser = build_arg_parser(
        "cloud_crash",
        default_n=200,
        default_dim=3,
        allowed_dims=(3,),
        default_mesh_mode="cartesian",
        default_rng_seed=20250429,
    )
    parser.add_argument("--n_clouds", type=int, default=8)
    args = parser.parse_args()

    create_cloud_crash(
        filename=resolve_filename(args, "cloud_crash"),
        num_seeds=args.n ** args.dimension,
        extent=args.extent,
        gamma=args.gamma,
        n_clouds=args.n_clouds,
        mesh_mode=args.mesh_mode,
        rng_seed=args.rng_seed,
    )
