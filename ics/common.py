"""
Common helper utilities for IC generation.

The framework supports two modes:

  - Serial (default): one Python process writes the whole IC file.
  - Parallel: under `mpirun -np N python create_X.py ...`, each rank generates
    its own slice of [row_lo, row_lo + n_local) particles and writes its slice
    via parallel HDF5. The Python script is the same — `write_ic()` auto-detects
    MPI at runtime and dispatches.

Existing IC scripts that haven't been ported keep working in serial via the
legacy `seed_positions()` helper. They just won't scale past one process.
"""

import argparse
import h5py
import numpy as np


_ALL_MESH_MODES = ("random", "cartesian", "polar_ring")


# ============================================================
# CLI helpers (unchanged)
# ============================================================

def build_arg_parser(
    name,
    *,
    description=None,
    default_n=64,
    default_dim=3,
    allowed_dims=(2, 3),
    default_mesh_mode="cartesian",
    allowed_mesh_modes=_ALL_MESH_MODES,
    default_perturbation=0.05,
    default_rng_seed=424242,
    default_extent=1.0,
    default_gamma=5.0 / 3.0,
):
    """Return an ArgumentParser with the shared IC-creation flags."""
    parser = argparse.ArgumentParser(description=description or f"Create {name} IC.")

    parser.add_argument(
        "--filename", type=str, default=None,
        help=f"output hdf5 path (default: IC_{name}_<D>D_N<n>.hdf5)",
    )
    parser.add_argument(
        "--n", type=int, default=default_n,
        help="cells per dimension (total seeds = n^dimension)",
    )
    parser.add_argument(
        "--dimension", type=int, default=default_dim, choices=list(allowed_dims),
    )
    parser.add_argument(
        "--mesh_mode", type=str, default=default_mesh_mode, choices=list(allowed_mesh_modes),
    )
    parser.add_argument(
        "--perturbation", type=float, default=default_perturbation,
        help="cartesian-grid jitter as fraction of cell size",
    )
    parser.add_argument("--rng_seed", type=int, default=default_rng_seed)
    parser.add_argument("--extent", type=float, default=default_extent)
    parser.add_argument("--gamma", type=float, default=default_gamma)

    return parser


def resolve_filename(args, name):
    """Pick args.filename if given, otherwise IC_<name>_<D>D_N<n>.hdf5."""
    if args.filename:
        return args.filename
    return f"IC_{name}_{args.dimension}D_N{args.n}.hdf5"


# ============================================================
# MPI runtime
# ============================================================

def mpi_runtime():
    """Return (comm, rank, nranks). (None, 0, 1) if mpi4py isn't loaded."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def even_split(N, P, i):
    """Same per-rank row range as proteus_mpi::decomp_even_split. (row_lo, row_hi)."""
    base, rem = divmod(N, P)
    lo = i * base + min(i, rem)
    hi = lo + base + (1 if i < rem else 0)
    return lo, hi


# ============================================================
# Stateless per-particle RNG (SplitMix64 → [0, 1))
# ============================================================

# Bit-deterministic and identical across launchers / nranks: given a fixed
# rng_seed, particle k's perturbation along axis d is the same value regardless
# of which rank materialised it. This replaces the stateful default_rng path
# the old serial scripts used.
def per_particle_uniform(rng_seed, particle_ids, axis):
    """Stateless uniform [0, 1) per (rng_seed, particle_id, axis)."""
    u64 = np.uint64
    pid = np.asarray(particle_ids, dtype=u64)

    # The multiplies wrap mod 2^64 by design (that's the whole point of the hash).
    # numpy 2.x warns on uint64 scalar overflow even when it's intentional, so
    # silence over/invalid for this block.
    with np.errstate(over="ignore", invalid="ignore"):
        state = (
            pid
            + u64(rng_seed & 0xFFFFFFFFFFFFFFFF) * u64(0x9E3779B97F4A7C15)
            + u64(axis & 0xFFFFFFFF) * u64(0xC2B2AE3D27D4EB4F)
        )

        # SplitMix64 finalizer
        state ^= state >> u64(30)
        state *= u64(0xBF58476D1CE4E5B9)
        state ^= state >> u64(27)
        state *= u64(0x94D049BB133111EB)
        state ^= state >> u64(31)

    return (state >> u64(11)).astype(np.float64) * (1.0 / (1 << 53))


def per_particle_signed(rng_seed, particle_ids, axis):
    """Stateless uniform [-1, 1) per (rng_seed, particle_id, axis). Convenience wrapper."""
    return 2.0 * per_particle_uniform(rng_seed, particle_ids, axis) - 1.0


# ============================================================
# Slice-aware position generation
# ============================================================

def seed_positions_slice(
    row_lo,
    n_local,
    total_num_seeds,
    dimension,
    extent=1.0,
    rng_seed=424242,
    mesh_mode="cartesian",
    perturbation=0.05,
):
    """Generate positions for global rows [row_lo, row_lo + n_local).

    "cartesian" and "random" modes are slice-deterministic: rank R's slice is
    a contiguous range of the global ordering, so the same (row_lo, n_local)
    range always yields the same positions.

    "polar_ring" mode is intentionally not slice-aware — its serial-only
    ring-walking algorithm doesn't decompose cleanly. It's only invoked from
    `seed_positions()` (serial fast path) below.
    """
    if mesh_mode not in _ALL_MESH_MODES:
        raise ValueError(f"Unknown mesh_mode '{mesh_mode}'")
    if mesh_mode == "polar_ring":
        raise NotImplementedError("polar_ring mode is serial-only; use the legacy seed_positions().")

    ids = np.arange(row_lo, row_lo + n_local, dtype=np.int64)

    if mesh_mode == "random":
        pos = np.empty((n_local, dimension), dtype=np.float64)
        for d in range(dimension):
            pos[:, d] = per_particle_uniform(rng_seed, ids, axis=d) * extent
        return pos

    # mesh_mode == "cartesian"
    if dimension == 2:
        nx = int(round(np.sqrt(total_num_seeds)))
        if nx * nx != total_num_seeds:
            raise ValueError("For cartesian 2D mesh, total_num_seeds must be a perfect square.")
        dx = extent / nx
        ix = ids % nx
        iy = ids // nx
        x = (ix.astype(np.float64) + 0.5) * dx
        y = (iy.astype(np.float64) + 0.5) * dx
        pos = np.column_stack((x, y))
    else:
        n = int(round(total_num_seeds ** (1.0 / 3.0)))
        if n * n * n != total_num_seeds:
            raise ValueError("For cartesian 3D mesh, total_num_seeds must be a perfect cube.")
        dx = extent / n
        # Ordering matches np.meshgrid(x1, y1, z1, indexing="xy") + column_stack(ravel())
        # used by the legacy seed_positions: ravel-order is (y, x, z) for indexing="xy"
        # in 3D meshgrid. Replicate that mapping from a flat index:
        #   k = iy * (n * n) + ix * n + iz
        iz = ids % n
        ix = (ids // n) % n
        iy = ids // (n * n)
        x = (ix.astype(np.float64) + 0.5) * dx
        y = (iy.astype(np.float64) + 0.5) * dx
        z = (iz.astype(np.float64) + 0.5) * dx
        pos = np.column_stack((x, y, z))

    # perturb the cartesian grid via stateless per-particle hash. Slice-safe.
    if perturbation != 0.0:
        for d in range(dimension):
            pos[:, d] += per_particle_signed(rng_seed, ids, axis=d) * (perturbation * dx)

    pos %= extent
    return pos


def seed_positions(
    num_seeds,
    dimension,
    extent=1.0,
    rng_seed=424242,
    mesh_mode="random",
    perturbation=0.05,
):
    """Serial fast path — generate all `num_seeds` positions at once.

    Cartesian and random modes route through seed_positions_slice (so serial and
    MPI agree bit-for-bit). polar_ring keeps its legacy serial-only algorithm.
    """
    if mesh_mode == "polar_ring":
        return _polar_ring_positions(num_seeds, dimension, extent, rng_seed)
    return seed_positions_slice(
        row_lo=0,
        n_local=num_seeds,
        total_num_seeds=num_seeds,
        dimension=dimension,
        extent=extent,
        rng_seed=rng_seed,
        mesh_mode=mesh_mode,
        perturbation=perturbation,
    )


# legacy polar_ring algorithm — preserved for the gresho-style 2D test ICs
def _polar_ring_positions(num_seeds, dimension, extent, rng_seed):
    rng = np.random.default_rng(rng_seed)
    pos = np.zeros((num_seeds, dimension), dtype=np.float64)
    seed_count = 0
    n_per_dim = int(round(num_seeds ** (1.0 / dimension)))
    d_ring = extent / n_per_dim
    extent_half = 0.5 * extent
    for ring_index in range(n_per_dim):
        n_cells_this_ring = max([1, int(round(2.0 * np.pi * ring_index))])
        phi = rng.uniform(0, 2.0 * np.pi)
        dphi = (2.0 * np.pi) / n_cells_this_ring
        for _i in range(n_cells_this_ring):
            radius = d_ring * ring_index
            x = radius * np.sin(phi)
            y = radius * np.cos(phi)
            if -extent_half <= x < extent_half and -extent_half <= y < extent_half:
                pos[seed_count] = [x, y]
                seed_count += 1
            if seed_count == num_seeds:
                break
            phi += dphi
    pos += 0.5 * extent
    return pos


# ============================================================
# HDF5 writer dispatch
# ============================================================

# Match the C++ reader's expected layout: header.dimension attr; mesh/pos
# (N×D doubles); hydro/{rho, vel, energy} (N or N×D doubles). Global IDs are
# assigned by the C++ side as row index in this file — no global_id dataset.
def write_ic(filename, n_global, dimension, fill_fn, args):
    """Serial-or-parallel IC writer.

    fill_fn(row_lo, n_local, args) must return (pos, vel, rho, energy) as
    numpy float64 arrays of shapes (n_local, D), (n_local, D), (n_local,), (n_local,).
    Per-particle state must only depend on the global row index for slice/serial
    agreement.
    """
    comm, rank, nranks = mpi_runtime()
    if nranks > 1:
        _check_parallel_h5py(comm)
        _write_parallel(filename, n_global, dimension, fill_fn, args, comm, rank, nranks)
    else:
        _write_serial(filename, n_global, dimension, fill_fn, args)


def _check_parallel_h5py(comm):
    if not h5py.get_config().mpi:
        msg = (
            "ICGEN: h5py was not built with MPI support. To generate ICs in parallel,\n"
            "       install a parallel-HDF5 + h5py build (HPC modules typically provide\n"
            "       these as `hdf5-mpi` + `h5py-parallel`, e.g.\n"
            "         brew install hdf5-mpi && pip install --no-binary h5py h5py\n"
            "       with CC=mpicc HDF5_MPI=ON). For small ICs that fit one process,\n"
            "       run without mpirun and the serial path will be used."
        )
        if comm is None or comm.Get_rank() == 0:
            print(msg)
        raise RuntimeError("h5py without MPI support; cannot write IC collectively.")


def _write_serial(filename, n_global, dimension, fill_fn, args):
    pos, vel, rho, energy = fill_fn(0, n_global, args)
    _validate_slice_shapes(pos, vel, rho, energy, n_global, dimension)
    with h5py.File(filename, "w") as f:
        _create_header(f, dimension)
        _create_datasets(f, n_global, dimension)
        f["mesh/pos"][...] = pos
        f["hydro/rho"][...] = rho
        f["hydro/vel"][...] = vel
        f["hydro/energy"][...] = energy
    print(f"ICGEN: wrote {filename}  ({n_global} cells, serial)")


def _write_parallel(filename, n_global, dimension, fill_fn, args, comm, rank, nranks):
    from mpi4py import MPI

    row_lo, row_hi = even_split(n_global, nranks, rank)
    n_local = row_hi - row_lo

    pos, vel, rho, energy = fill_fn(row_lo, n_local, args)
    _validate_slice_shapes(pos, vel, rho, energy, n_local, dimension)

    if rank == 0:
        print(f"ICGEN: writing {filename}  ({n_global} cells, parallel, {nranks} ranks)")

    with h5py.File(filename, "w", driver="mpio", comm=comm) as f:
        _create_header(f, dimension)
        _create_datasets(f, n_global, dimension)
        # Collective writes — each rank writes its own contiguous row range.
        with f["mesh/pos"].collective:
            f["mesh/pos"][row_lo:row_hi, :] = pos
        with f["hydro/rho"].collective:
            f["hydro/rho"][row_lo:row_hi] = rho
        with f["hydro/vel"].collective:
            f["hydro/vel"][row_lo:row_hi, :] = vel
        with f["hydro/energy"].collective:
            f["hydro/energy"][row_lo:row_hi] = energy

    comm.Barrier()
    if rank == 0:
        print(f"ICGEN: done writing {filename}")


def _validate_slice_shapes(pos, vel, rho, energy, n_expected, dimension):
    if pos.shape != (n_expected, dimension):
        raise ValueError(f"pos shape {pos.shape} != ({n_expected}, {dimension})")
    if vel.shape != (n_expected, dimension):
        raise ValueError(f"vel shape {vel.shape} != ({n_expected}, {dimension})")
    if rho.shape != (n_expected,):
        raise ValueError(f"rho shape {rho.shape} != ({n_expected},)")
    if energy.shape != (n_expected,):
        raise ValueError(f"energy shape {energy.shape} != ({n_expected},)")


def _create_header(f, dimension):
    hdr = f.create_group("header")
    hdr.attrs["dimension"] = dimension


def _create_datasets(f, n_global, dimension):
    f.create_group("mesh")
    f.create_group("hydro")

    # Chunk row count chosen so each chunk is ~1-2 MB; HDF5-parallel writes
    # scale much better with chunked layout than contiguous.
    target_chunk_bytes = 1 << 20  # 1 MB
    pos_row_bytes = dimension * 8
    chunk_rows = max(1, min(n_global, target_chunk_bytes // pos_row_bytes))

    f.create_dataset(
        "mesh/pos", shape=(n_global, dimension), dtype="f8",
        chunks=(chunk_rows, dimension),
    )
    f.create_dataset(
        "hydro/rho", shape=(n_global,), dtype="f8",
        chunks=(chunk_rows,),
    )
    f.create_dataset(
        "hydro/vel", shape=(n_global, dimension), dtype="f8",
        chunks=(chunk_rows, dimension),
    )
    f.create_dataset(
        "hydro/energy", shape=(n_global,), dtype="f8",
        chunks=(chunk_rows,),
    )


def collective_sum_int(comm, x):
    """Sum a single int across ranks via mpi4py. Falls back to local value in serial."""
    if comm is None:
        return int(x)
    from mpi4py import MPI
    return int(comm.allreduce(int(x), op=MPI.SUM))
