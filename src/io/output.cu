#include "../global/allvars.h"
#include "../mpi/decomp.h"
#include "../mpi/mpi_compat.h"
#include "../voronoi/voronoi.h"
#include "output.h"
#include "profiler/profiler.h"
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

static void write_attr_int(hid_t group, const char* name, int value);
static void write_attr_double(hid_t group, const char* name, double value);
static bool write_dataset_1d(hid_t parent, const char* name, const double* data, hsize_t n);
static bool write_dataset_2d(hid_t parent, const char* name, const double* data, hsize_t n, hsize_t dim);

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
    Profiler::StartTimer("MPI_WAIT");
    MPI_Barrier(proteus_mpi::decomp.cart_comm);
    Profiler::EndTimer("MPI_WAIT");
#endif
    return ok;
}

// ============================================================
// write snapshot
// ============================================================

void OutputHandler::snapshot(int snap_num, VMesh* mesh, const hydro::primvars* primvar, double t_sim, int step) {
    Profiler::StartTimer("SNAPSHOTS");

    const int n_hydro  = (int)mesh->n_hydro;
    const int nranks   = proteus_mpi::nranks();
    const int rank     = proteus_mpi::rank();
    const int n_global = logging::sum_global(n_hydro);

    // file-per-rank under multi-rank
    std::string output_file = "snapshot_" + std::to_string(snap_num);
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
    write_attr_double(header_group, "extent", 1.0);
    write_attr_double(header_group, "time", t_sim);
    write_attr_int(header_group, "step", step);
    write_attr_int(header_group, "n_global", n_global);
    write_attr_int(header_group, "nranks", nranks);
    write_attr_int(header_group, "rank", rank);
    H5Gclose(header_group);

    // flatten seeds (mesh->seeds always 3D, we store 2D/3D depending on DIMENSION)
    std::vector<double> pos_flat(n_hydro * DIMENSION);
    for (int i = 0; i < n_hydro; i++) {
        pos_flat[i * DIMENSION + 0] = mesh->seeds[i].x;
        pos_flat[i * DIMENSION + 1] = mesh->seeds[i].y;
#ifdef dim_3D
        pos_flat[i * DIMENSION + 2] = mesh->seeds[i].z;
#endif
    }

    // write mesh/pos
    hid_t mesh_group = H5Gcreate(file_id, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (!write_dataset_2d(mesh_group, "pos", pos_flat.data(), n_hydro, DIMENSION)) {
        H5Gclose(mesh_group);
        H5Fclose(file_id);
        exit(EXIT_FAILURE);
    }
    H5Gclose(mesh_group);

    // write hydro/{rho,vel,energy}
    hid_t hydro_group = H5Gcreate(file_id, "hydro", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (!write_dataset_1d(hydro_group, "rho", primvar->rho, n_hydro) ||
        !write_dataset_2d(hydro_group, "vel", reinterpret_cast<const double*>(primvar->v), n_hydro, DIMENSION) ||
        !write_dataset_1d(hydro_group, "energy", primvar->E, n_hydro)) {
        H5Gclose(hydro_group);
        H5Fclose(file_id);
        exit(EXIT_FAILURE);
    }
    H5Gclose(hydro_group);
    H5Fclose(file_id);

    Profiler::EndTimer("SNAPSHOTS");
}

// ============================================================
// per-step log line
// ============================================================

// prints current step, t, dt and ETA
void print_log(
    int step, std::chrono::steady_clock::time_point wall, double t_sim, double dt, double t_start, double t_end) {

    const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall).count();
    logging::root() << "\nSIM: Step " << step << "  t = " << t_sim << "  dt = " << dt << "  ETA = "
                    << format_hms((t_sim > t_start) ? elapsed_s * (t_end - t_sim) / (t_sim - t_start) : 0.0)
                    << std::endl;
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
