#include "../global/allvars.h"
#include "../mpi/mpi_compat.h"
#include "../mpi/rebalance.h"
#include "../voronoi/voronoi.h"
#include "output.h"
#include "profiler/profiler.h"
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

static void write_attr_int(hid_t group, const char* name, int value);
static void write_attr_int64(hid_t group, const char* name, int64_t value);
static void write_attr_double(hid_t group, const char* name, double value);
static bool write_dataset_1d(hid_t parent, const char* name, const double* data, hsize_t n);
static bool write_dataset_2d(hid_t parent, const char* name, const double* data, hsize_t n, hsize_t dim);
#ifdef OUTPUT_MESH
static bool write_dataset_1d_i32(hid_t parent, const char* name, const int* data, hsize_t n);
static bool write_dataset_1d_i64(hid_t parent, const char* name, const int64_t* data, hsize_t n);
static void write_mesh_geometry(hid_t mesh_group, int n_hydro);
#endif

OutputHandler::OutputHandler(const std::string& outputDir) : outputDirectory(outputDir) {}

// ============================================================
// init
// ============================================================

bool OutputHandler::initialize() {
    // create output directory on rank 0 only
    bool ok = true;
    if (proteus_mpi::is_root()) {
        struct stat st;
        if (stat(outputDirectory.c_str(), &st) != 0) {
            if (mkdir(outputDirectory.c_str(), 0755) != 0) {
                std::cerr << "OUTPUT: Error! Could not create output directory: " << outputDirectory << std::endl;
                ok = false;
            } else {
                logging::root() << "OUTPUT: Created new output directory: " << outputDirectory << std::endl;
            }
        }
        if (ok) logging::root() << "OUTPUT: directory: " << outputDirectory << std::endl;
    }
#ifdef USE_MPI
    {
        PROFILE_MPI("OUTPUT_INIT_BARRIER");
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
    return ok;
}

// ============================================================
// write snapshot
// ============================================================

void OutputHandler::write_snapshot() {
    PROFILE("IO_SNAPSHOT");

    const int     n_hydro  = (int)sim.mesh->n_hydro;
    const int     nranks   = proteus_mpi::nranks();
    const int     rank     = proteus_mpi::rank();
    const int64_t n_global = logging::sum_global((long long)n_hydro);

    // file-per-rank under multi-rank
    std::string output_file = "snapshot_" + std::to_string(sim.snap_num);
    if (nranks > 1) output_file += "." + std::to_string(rank);
    output_file += ".hdf5";
    const std::string fullPath = outputDirectory + output_file;

    if (nranks > 1) {
        std::cout << "OUTPUT: Writing snapshot to: " << fullPath << std::endl;
    } else {
        logging::root() << "OUTPUT: Writing snapshot to: " << fullPath << std::endl;
    }

    hid_t file_id = H5Fcreate(fullPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "OUTPUT: Error! Could not create HDF5 file: " << fullPath << std::endl;
        exit(EXIT_FAILURE);
    }

    // write header
    hid_t header_group = H5Gcreate(file_id, "header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_attr_int(header_group, "dimension", DIMENSION);
    write_attr_double(header_group, "time", sim.t_sim);
    write_attr_int(header_group, "step", sim.step);
    write_attr_int64(header_group, "n_global", n_global);
    write_attr_int(header_group, "nranks", nranks);
    write_attr_int(header_group, "rank", rank);

#ifdef ASTRO_PHYSICS
    // code-unit base factors, so analysis can convert snapshots back to cgs
    write_attr_double(header_group, "UnitLength_in_cm", units.UnitLength_in_cm);
    write_attr_double(header_group, "UnitMass_in_g", units.UnitMass_in_g);
    write_attr_double(header_group, "UnitVelocity_in_cm_per_s", units.UnitVelocity_in_cm_per_s);
#endif

    // /header/profiler: one double attr per timer with this rank's cum seconds.
    // Restart reads these to seed Profiler::m_Timings directly — no more reliance
    // on a text profile log for restart state.
    hid_t prof_group = H5Gcreate(header_group, "profiler", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for (const auto& kv : Profiler::CurrentCumulative()) {
        write_attr_double(prof_group, kv.first.c_str(), kv.second);
    }
    H5Gclose(prof_group);

    H5Gclose(header_group);

    // flatten seeds (mesh->seeds always 3D, we store 2D/3D depending on DIMENSION)
    std::vector<double> pos_flat(n_hydro * DIMENSION);
    for (int i = 0; i < n_hydro; i++) {
        pos_flat[i * DIMENSION + 0] = sim.mesh->seeds[i].x;
        pos_flat[i * DIMENSION + 1] = sim.mesh->seeds[i].y;
#ifdef dim_3D
        pos_flat[i * DIMENSION + 2] = sim.mesh->seeds[i].z;
#endif
    }

    // write mesh/pos
    hid_t mesh_group = H5Gcreate(file_id, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (!write_dataset_2d(mesh_group, "pos", pos_flat.data(), n_hydro, DIMENSION)) {
        H5Gclose(mesh_group);
        H5Fclose(file_id);
        exit(EXIT_FAILURE);
    }

#ifdef OUTPUT_MESH
    // cell volumes alongside the seeds. The tessellation is a deterministic function of the
    // seed positions, so this is recoverable after the fact, but storing it saves analysis
    // code from rebuilding the mesh or approximating volumes with a k-NN estimator.
    if (!write_dataset_1d(mesh_group, "volume", sim.mesh->volumes, n_hydro)) {
        H5Gclose(mesh_group);
        H5Fclose(file_id);
        exit(EXIT_FAILURE);
    }

    // Full Voronoi-geometry dump for the mesh-generation verification test (compared against
    // an independent tessellation). Read host-side from the managed VMesh arrays — no
    // Voronoi-kernel change, so the build codegen is untouched.
    write_mesh_geometry(mesh_group, n_hydro);
#endif

    H5Gclose(mesh_group);

    // write hydro/{rho,vel,energy}
    hid_t hydro_group = H5Gcreate(file_id, "hydro", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (!write_dataset_1d(hydro_group, "rho", sim.primvar->rho, n_hydro) ||
        !write_dataset_2d(hydro_group, "vel", reinterpret_cast<const double*>(sim.primvar->v), n_hydro, DIMENSION) ||
        !write_dataset_1d(hydro_group, "energy", sim.primvar->E, n_hydro)) {
        H5Gclose(hydro_group);
        H5Fclose(file_id);
        exit(EXIT_FAILURE);
    }
    H5Gclose(hydro_group);
    H5Fclose(file_id);

    sim.snap_num += 1;
    // Skip the advance for the initial t=0 snapshot: begrun already set
    // t_nextoutput = t_sim + output_dt, so advancing here too would put the first
    // cadenced snapshot at 2*output_dt. The test runs post-increment, so snap_0 is 1.
    if (sim.snap_num != 1) { sim.t_nextoutput += sim.output_dt; }
}

// ============================================================
// per-step log line
// ============================================================

// prints all per-step diagnostics: step, t, dt, ETA and the load-imbalance probe
void print_log() {

    const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - sim.wall_start).count();
    logging::root() << "\nSIM: Step " << sim.step << "  t = " << sim.t_sim << "  dt = " << *sim.dt << "  ETA = "
                    << format_hms((sim.t_sim > sim.t_start)
                                      ? elapsed_s * (sim.t_end - sim.t_sim) / (sim.t_sim - sim.t_start)
                                      : 0.0)
                    << std::endl;

    // load-imbalance probe (no-op in serial / on non-probe steps)
    proteus_mpi::rebalance_imbalance_log(sim.step, sim.mesh);
}

// ============================================================
// helpers
// ============================================================

// HDF5 write operations
static void write_attr_int(hid_t group, const char* name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t a     = H5Acreate(group, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_INT, &value);
    H5Aclose(a);
    H5Sclose(space);
}

static void write_attr_int64(hid_t group, const char* name, int64_t value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t a     = H5Acreate(group, name, H5T_NATIVE_INT64, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_INT64, &value);
    H5Aclose(a);
    H5Sclose(space);
}

static void write_attr_double(hid_t group, const char* name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t a     = H5Acreate(group, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(a);
    H5Sclose(space);
}

static bool write_dataset_1d(hid_t parent, const char* name, const double* data, hsize_t n) {
    hsize_t dims[1] = {n};
    hid_t   space   = H5Screate_simple(1, dims, NULL);
    hid_t   dset    = H5Dcreate(parent, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) {
        std::cerr << "OUTPUT: Error! Could not create dataset '" << name << "'" << std::endl;
        H5Sclose(space);
        return false;
    }
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dset);
    H5Sclose(space);
    return true;
}

static bool write_dataset_2d(hid_t parent, const char* name, const double* data, hsize_t n, hsize_t dim) {
    hsize_t dims[2] = {n, dim};
    hid_t   space   = H5Screate_simple(2, dims, NULL);
    hid_t   dset    = H5Dcreate(parent, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) {
        std::cerr << "OUTPUT: Error! Could not create dataset '" << name << "'" << std::endl;
        H5Sclose(space);
        return false;
    }
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dset);
    H5Sclose(space);
    return true;
}

// OUTPUT_MESH: full Voronoi-geometry dump (verification only)
// ============================================================

#ifdef OUTPUT_MESH

static bool write_dataset_1d_i32(hid_t parent, const char* name, const int* data, hsize_t n) {
    hsize_t dims[1] = {n};
    hid_t   space   = H5Screate_simple(1, dims, NULL);
    hid_t   dset    = H5Dcreate(parent, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) {
        std::cerr << "OUTPUT: Error! Could not create dataset '" << name << "'" << std::endl;
        H5Sclose(space);
        return false;
    }
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dset);
    H5Sclose(space);
    return true;
}

static bool write_dataset_1d_i64(hid_t parent, const char* name, const int64_t* data, hsize_t n) {
    hsize_t dims[1] = {n};
    hid_t   space   = H5Screate_simple(1, dims, NULL);
    hid_t   dset    = H5Dcreate(parent, name, H5T_NATIVE_INT64, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) {
        std::cerr << "OUTPUT: Error! Could not create dataset '" << name << "'" << std::endl;
        H5Sclose(space);
        return false;
    }
    H5Dwrite(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dset);
    H5Sclose(space);
    return true;
}

// Compact the ragged per-face SoA (neighbor_cell / face_area, sliced by face_ptr/face_counts)
// into CSR arrays and write the whole mesh geometry into `mesh_group`, in ascending-k order:
//   mesh/n_faces       [N]      : faces of cell k
//   mesh/face_offset   [N+1]    : CSR offsets, so cell k owns faces [off[k], off[k+1])
//   mesh/face_neighbor [F]      : neighbor cell id (real k), or -1 for a box-boundary face
//   mesh/face_area     [F]      : Voronoi facet area (2D: edge length)
//   mesh/face_normal   [F,D]    : unit perpendicular-bisector normal, seed_k -> seed_neighbor
//   mesh/centroid      [N,D]    : volume-weighted cell centroid (VMesh::com)
// mesh/volume [N] and mesh/pos [N,D] are written by the caller. The face normal is the
// minimum-image seed-to-seed direction on the unit box — exactly how the build defines a
// face plane — reconstructed here so the register-heavy Voronoi kernel is left untouched.
static void write_mesh_geometry(hid_t mesh_group, int n_hydro) {
    const VMesh* m = sim.mesh;

    int64_t F = 0;
    for (int k = 0; k < n_hydro; k++) { F += (int64_t)m->face_counts[k]; }

    std::vector<int>     n_faces((size_t)n_hydro);
    std::vector<int64_t> face_offset((size_t)n_hydro + 1);
    std::vector<int>     face_neighbor((size_t)F);
    std::vector<double>  face_area((size_t)F);
    std::vector<double>  face_normal((size_t)F * DIMENSION);
    std::vector<double>  com_flat((size_t)n_hydro * DIMENSION);

    int64_t run = 0;
    for (int k = 0; k < n_hydro; k++) {
        const hsize_t cnt = m->face_counts[k];
        const hsize_t ptr = m->face_ptr[k];
        n_faces[k]        = (int)cnt;
        face_offset[k]    = run;

        const double3 sk                    = m->seeds[k];
        com_flat[(size_t)k * DIMENSION + 0] = m->com[k].x;
        com_flat[(size_t)k * DIMENSION + 1] = m->com[k].y;
#ifdef dim_3D
        com_flat[(size_t)k * DIMENSION + 2] = m->com[k].z;
#endif
        for (hsize_t f = 0; f < cnt; f++) {
            const hsize_t src  = ptr + f;
            const int     nbr  = m->neighbor_cell[src];
            const size_t  dst  = (size_t)(run + (int64_t)f);
            face_neighbor[dst] = nbr;
            face_area[dst]     = m->face_area[src];

            double nx = 0.0, ny = 0.0, nz = 0.0;
            if (nbr >= 0) {
                const double3 sn = m->seeds[nbr];
                double        dx = sn.x - sk.x;
                double        dy = sn.y - sk.y;
                dx -= std::round(dx); // minimum image on the unit box
                dy -= std::round(dy);
                double dz = 0.0;
#ifdef dim_3D
                dz = sn.z - sk.z;
                dz -= std::round(dz);
#endif
                const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double inv = (len > 0.0) ? 1.0 / len : 0.0;
                nx = dx * inv;
                ny = dy * inv;
                nz = dz * inv;
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
    ok = ok && write_dataset_1d_i32(mesh_group, "n_faces", n_faces.data(), (hsize_t)n_hydro);
    ok = ok && write_dataset_1d_i64(mesh_group, "face_offset", face_offset.data(), (hsize_t)n_hydro + 1);
    ok = ok && write_dataset_1d_i32(mesh_group, "face_neighbor", face_neighbor.data(), (hsize_t)F);
    ok = ok && write_dataset_1d(mesh_group, "face_area", face_area.data(), (hsize_t)F);
    ok = ok && write_dataset_2d(mesh_group, "face_normal", face_normal.data(), (hsize_t)F, DIMENSION);
    ok = ok && write_dataset_2d(mesh_group, "centroid", com_flat.data(), (hsize_t)n_hydro, DIMENSION);
    if (!ok) {
        std::cerr << "OUTPUT: Error! failed to write OUTPUT_MESH geometry" << std::endl;
        exit(EXIT_FAILURE);
    }

    logging::root() << "OUTPUT: wrote full mesh geometry (" << n_hydro << " cells, " << F << " faces)" << std::endl;
}

#endif // OUTPUT_MESH
