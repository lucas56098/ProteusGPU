// implements OutputHandler + print_log (output.h)

#include "../global/allvars.h"
#include "../mpi/decomp.h"
#include "../mpi/mpi_compat.h"
#include "../mpi/rebalance.h"
#include "../voronoi/voronoi.h"
#include "h5.h"
#include "output.h"
#include "profiler/profiler.h"
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#ifdef OUTPUT_MESH
static bool write_mesh_geometry(hid_t mesh_group, int n_hydro);
#endif

OutputHandler::OutputHandler(const std::string& output_dir) : output_directory(output_dir) {}

// ==========================================================
// snapshots
// ==========================================================

// creates output_directory if it does not exist (rank 0 does it)
bool OutputHandler::initialize() {
    bool ok = true;
    if (proteus_mpi::is_root()) {
        struct stat st;
        if (stat(output_directory.c_str(), &st) != 0) {
            if (mkdir(output_directory.c_str(), 0755) != 0) {
                std::cerr << "OUTPUT: Error! Could not create output directory: " << output_directory << std::endl;
                ok = false;
            } else {
                logging::root() << "OUTPUT: Created new output directory: " << output_directory << std::endl;
            }
        }
        if (ok) logging::root() << "OUTPUT: directory: " << output_directory << std::endl;
    }
#ifdef USE_MPI
    {
        // no rank may write before the directory is there
        PROFILE_MPI("OUTPUT_INIT_BARRIER");
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
    return ok;
}

// writes one whole snapshot file
static bool write_snapshot_file(const std::string& path, int n_hydro, int nranks, int rank, int64_t n_global) {

    h5::File file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    if (!file.valid()) {
        std::cerr << "OUTPUT: Error! Could not create HDF5 file: " << path << std::endl;
        return false;
    }

    {
        // header: dimension, time, step, cell counts and the ngb grid size
        h5::Group header_group(H5Gcreate(file, "header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if (!h5::write_attr(header_group, "dimension", DIMENSION) || !h5::write_attr(header_group, "time", sim.t_sim) ||
            !h5::write_attr(header_group, "step", sim.step) || !h5::write_attr(header_group, "n_global", n_global) ||
            !h5::write_attr(header_group, "nranks", nranks) || !h5::write_attr(header_group, "rank", rank) ||
            !h5::write_attr(header_group, "knn_N_grid", sim.mesh->knn->N_grid)) {
            return false;
        }

#ifdef ASTRO_PHYSICS
        // unit system, needed to read the snapshot back in cgs
        if (!h5::write_attr(header_group, "UnitLength_in_cm", units.UnitLength_in_cm) ||
            !h5::write_attr(header_group, "UnitMass_in_g", units.UnitMass_in_g) ||
            !h5::write_attr(header_group, "UnitVelocity_in_cm_per_s", units.UnitVelocity_in_cm_per_s)) {
            return false;
        }
#endif

        // cumulative timer values, a restart picks them up again
        h5::Group prof_group(H5Gcreate(header_group, "profiler", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        for (const auto& kv : Profiler::current_cumulative()) {
            if (!h5::write_attr(prof_group, kv.first.c_str(), kv.second)) { return false; }
        }
    }

#ifdef USE_MPI
    {
        // split tables of this run, a restart has to start from the same bricks
        const auto& dc = proteus_mpi::decomp;
        h5::Group   decomp_group(H5Gcreate(file, "decomp", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if (!h5::write_dataset_1d(decomp_group, "splits_x", dc.splits[0], (hsize_t)(dc.dims[0] + 1)) ||
            !h5::write_dataset_1d(decomp_group, "splits_y", dc.splits[1], (hsize_t)(dc.dims[1] + 1)) ||
            !h5::write_dataset_1d(decomp_group, "splits_z", dc.splits[2], (hsize_t)(dc.dims[2] + 1))) {
            return false;
        }
    }
#endif

    // seeds as flat rows of DIM values
    std::vector<double> pos_flat(n_hydro * DIMENSION);
    for (int i = 0; i < n_hydro; i++) {
        pos_flat[i * DIMENSION + 0] = sim.mesh->seeds[i].x;
        pos_flat[i * DIMENSION + 1] = sim.mesh->seeds[i].y;
#ifdef dim_3D
        pos_flat[i * DIMENSION + 2] = sim.mesh->seeds[i].z;
#endif
    }

    {
        // mesh: seed positions, v_mesh (a restart needs it), volumes and faces with OUTPUT_MESH
        h5::Group mesh_group(H5Gcreate(file, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if (!h5::write_dataset_2d(mesh_group, "pos", pos_flat.data(), n_hydro, DIMENSION)) { return false; }

#ifdef MOVING_MESH
        std::vector<double> vmesh_flat(n_hydro * DIMENSION);
        for (int i = 0; i < n_hydro; i++) {
            vmesh_flat[i * DIMENSION + 0] = sim.mesh->v_mesh[i].x;
            vmesh_flat[i * DIMENSION + 1] = sim.mesh->v_mesh[i].y;
#ifdef dim_3D
            vmesh_flat[i * DIMENSION + 2] = sim.mesh->v_mesh[i].z;
#endif
        }
        if (!h5::write_dataset_2d(mesh_group, "v_mesh", vmesh_flat.data(), n_hydro, DIMENSION)) { return false; }
#endif

#ifdef OUTPUT_MESH

        if (!h5::write_dataset_1d(mesh_group, "volume", sim.mesh->volumes, n_hydro)) { return false; }
        if (!write_mesh_geometry(mesh_group, n_hydro)) { return false; }
#endif
    }

    {
        // hydro: rho, vel and energy (total energy per volume)
        h5::Group hydro_group(H5Gcreate(file, "hydro", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if (!h5::write_dataset_1d(hydro_group, "rho", sim.primvar->rho, n_hydro) ||
            !h5::write_dataset_2d(
                hydro_group, "vel", reinterpret_cast<const double*>(sim.primvar->v), n_hydro, DIMENSION) ||
            !h5::write_dataset_1d(hydro_group, "energy", sim.primvar->E, n_hydro)) {
            return false;
        }
    }

    return true;
}

// writes this rank's snapshot and sets the time of the next one
void OutputHandler::write_snapshot() {
    PROFILE("IO_SNAPSHOT");

    const int n_hydro = (int)sim.mesh->n_hydro;
    const int nranks  = proteus_mpi::nranks();
    const int rank    = proteus_mpi::rank();
    // total cells over all ranks
    const int64_t n_global = logging::sum_global((long long)n_hydro);

    // one file per rank as soon as there is more than one
    std::string output_file = "snapshot_" + std::to_string(sim.snap_num);
    if (nranks > 1) output_file += "." + std::to_string(rank);
    output_file += ".hdf5";
    const std::string full_path = output_directory + output_file;

    if (nranks > 1) {
        std::cout << "OUTPUT: Writing snapshot to: " << full_path << std::endl;
    } else {
        logging::root() << "OUTPUT: Writing snapshot to: " << full_path << std::endl;
    }

    // abort only out here: exit() skips destructors, so the h5 handles must be closed first
    if (!write_snapshot_file(full_path, n_hydro, nranks, rank, n_global)) {
        proteus_mpi::exit_failure("OUTPUT: failed to write snapshot %s\n", full_path.c_str());
    }

    sim.snap_num += 1;
    // snapshot_0 is written before the loop, begrun already set the time of the next output
    if (sim.snap_num != 1) { sim.t_nextoutput += sim.output_dt; }
}

// ==========================================================
// per-step log
// ==========================================================

// one line per step: step, t, dt and the ETA from the wall time so far
void print_log() {

    const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - sim.wall_start).count();
    logging::root() << "\nSIM: Step " << sim.step << "  t = " << sim.t_sim << "  dt = " << *sim.dt << "  ETA = "
                    << format_hms((sim.t_sim > sim.t_start)
                                      ? elapsed_s * (sim.t_end - sim.t_sim) / (sim.t_sim - sim.t_start)
                                      : 0.0)
                    << std::endl;

    // prints the load imbalance every imbalance_log_interval steps
    proteus_mpi::rebalance_imbalance_log(sim.step, sim.mesh);
}

#ifdef OUTPUT_MESH

// ==========================================================
// full mesh geometry (OUTPUT_MESH)
// ==========================================================

// writes the mesh itself for analysis: faces per cell, neighbours, areas, normals and centroids
static bool write_mesh_geometry(hid_t mesh_group, int n_hydro) {
    const VMesh* m = sim.mesh;

    // total faces on this rank
    int64_t F = 0;
    for (int k = 0; k < n_hydro; k++) {
        F += (int64_t)m->face_counts[k];
    }

    std::vector<int>     n_faces((size_t)n_hydro);
    std::vector<int64_t> face_offset((size_t)n_hydro + 1);
    std::vector<int>     face_neighbor((size_t)F);
    std::vector<double>  face_area((size_t)F);
    std::vector<double>  face_normal((size_t)F * DIMENSION);
    std::vector<double>  com_flat((size_t)n_hydro * DIMENSION);

    // pack the per-cell face lists into flat arrays
    int64_t run = 0;
    for (int k = 0; k < n_hydro; k++) {
        const uint64_t cnt = m->face_counts[k];
        const uint64_t ptr = m->face_ptr[k];
        n_faces[k]         = (int)cnt;
        face_offset[k]     = run;

        const double3 sk                    = m->seeds[k];
        com_flat[(size_t)k * DIMENSION + 0] = m->com[k].x;
        com_flat[(size_t)k * DIMENSION + 1] = m->com[k].y;
#ifdef dim_3D
        com_flat[(size_t)k * DIMENSION + 2] = m->com[k].z;
#endif
        for (uint64_t f = 0; f < cnt; f++) {
            const uint64_t src = ptr + f;
            const int      nbr = m->neighbor_cell[src];
            const size_t   dst = (size_t)(run + (int64_t)f);
            face_neighbor[dst] = nbr;
            face_area[dst]     = m->face_area[src];

            double nx = 0.0, ny = 0.0;
#ifdef dim_3D
            double nz = 0.0;
#endif
            // face normal = direction to the neighbour seed (periodic wrap), 0 on a box wall face
            if (nbr >= 0) {
                const double3 sn = get_seed_at(nbr, n_hydro, m);
                double        dx = sn.x - sk.x;
                double        dy = sn.y - sk.y;
                dx -= std::round(dx);
                dy -= std::round(dy);
                double dz = 0.0;
#ifdef dim_3D
                dz = sn.z - sk.z;
                dz -= std::round(dz);
#endif
                const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double inv = (len > 0.0) ? 1.0 / len : 0.0;
                nx               = dx * inv;
                ny               = dy * inv;
#ifdef dim_3D
                nz = dz * inv;
#endif
            }
            face_normal[dst * DIMENSION + 0] = nx;
            face_normal[dst * DIMENSION + 1] = ny;
#ifdef dim_3D
            face_normal[dst * DIMENSION + 2] = nz;
#endif
        }
        run += (int64_t)cnt;
    }
    face_offset[(size_t)n_hydro] = run;

    bool ok = true;
    // write the arrays
    ok = ok && h5::write_dataset_1d(mesh_group, "n_faces", n_faces.data(), (hsize_t)n_hydro);
    ok = ok && h5::write_dataset_1d(mesh_group, "face_offset", face_offset.data(), (hsize_t)n_hydro + 1);
    ok = ok && h5::write_dataset_1d(mesh_group, "face_neighbor", face_neighbor.data(), (hsize_t)F);
    ok = ok && h5::write_dataset_1d(mesh_group, "face_area", face_area.data(), (hsize_t)F);
    ok = ok && h5::write_dataset_2d(mesh_group, "face_normal", face_normal.data(), (hsize_t)F, DIMENSION);
    ok = ok && h5::write_dataset_2d(mesh_group, "centroid", com_flat.data(), (hsize_t)n_hydro, DIMENSION);
    if (!ok) {
        std::cerr << "OUTPUT: Error! failed to write OUTPUT_MESH geometry" << std::endl;
        return false;
    }

    logging::root() << "OUTPUT: wrote full mesh geometry (" << n_hydro << " cells, " << F << " faces)" << std::endl;
    return true;
}

#endif
