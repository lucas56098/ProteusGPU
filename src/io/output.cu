#include "output.h"
#include "../global/allvars.h"
#include "../voronoi/voronoi.h"
#include "profiler/profiler.h"
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

OutputHandler::OutputHandler(const std::string& outputDir) : outputDirectory(outputDir) {}

bool OutputHandler::initialize() {
    // create output directory if it doesn't exist
    struct stat st;
    if (stat(outputDirectory.c_str(), &st) != 0) {
        if (mkdir(outputDirectory.c_str(), 0755) != 0) {
            std::cerr << "OUTPUT: Error! Could not create output directory: " << outputDirectory << std::endl;
            return false;
        }
        std::cout << "OUTPUT: Created new output directory: " << outputDirectory << std::endl;
    }
    std::cout << "OUTPUT: directory: " << outputDirectory << std::endl;

    return true;
}

// wrapper to convert Vmesh to meshData and then write the snapshot file
void OutputHandler::snapshot(int snap_num, VMesh* mesh, const hydro::primvars* primvar, int n_hydro, double t_sim) {
    PROFILE_START("SNAPSHOTS");

    MeshCellData meshData;
    vmesh_to_meshdata(mesh, meshData);

    std::vector<unsigned int> original_to_current(n_hydro);
    for (int k = 0; k < n_hydro; k++) {
        original_to_current[mesh->cell_to_original[k]] = (unsigned int)k;
    }
    std::vector<double>     rho_out(n_hydro);
    std::vector<POINT_TYPE> vel_out(n_hydro);
    std::vector<double>     E_out(n_hydro);
    if (primvar) {
        for (int file_id = 0; file_id < n_hydro; file_id++) {
            unsigned int k = original_to_current[file_id];
            if (primvar->rho) rho_out[file_id] = primvar->rho[k];
            if (primvar->v)   vel_out[file_id] = primvar->v[k];
            if (primvar->E)   E_out[file_id]   = primvar->E[k];
        }
    }
    hydro::primvars primvar_inv;
    primvar_inv.rho = primvar ? rho_out.data() : nullptr;
    primvar_inv.v   = primvar ? vel_out.data() : nullptr;
    primvar_inv.E   = primvar ? E_out.data()   : nullptr;

    std::string output_file = "snapshot_" + std::to_string(snap_num) + ".hdf5";

    if (!writeSnapshot(output_file, meshData, &primvar_inv, n_hydro, t_sim)) { exit(EXIT_FAILURE); }

    PROFILE_END("SNAPSHOTS");
}

void OutputHandler::vmesh_to_meshdata(VMesh* mesh, MeshCellData& meshData) {
    int n_pts = (int)mesh->n_hydro;

    // header
    meshData.header.dimension = DIMENSION;
    meshData.header.extent    = 1.0;
    meshData.header.n         = n_pts;
    meshData.header.k         = _K_;
    meshData.header.nmax      = _MAX_P_;
    meshData.header.seed      = 0;

    meshData.seeds_dims = {(hsize_t)n_pts, DIMENSION};

    // file_id -> current k (inverse of cell_to_original)
    std::vector<unsigned int> original_to_current(n_pts);
    for (int k = 0; k < n_pts; k++) {
        original_to_current[mesh->cell_to_original[k]] = (unsigned int)k;
    }

    meshData.seeds.resize(n_pts * DIMENSION);
    for (int file_id = 0; file_id < n_pts; file_id++) {
        unsigned int k = original_to_current[file_id];
        meshData.seeds[file_id * DIMENSION + 0] = mesh->seeds[k].x;
        meshData.seeds[file_id * DIMENSION + 1] = mesh->seeds[k].y;
#ifdef dim_3D
        meshData.seeds[file_id * DIMENSION + 2] = mesh->seeds[k].z;
#endif
    }

    meshData.volumes.resize(n_pts);
    for (int file_id = 0; file_id < n_pts; file_id++) {
        meshData.volumes[file_id] = mesh->volumes[original_to_current[file_id]];
    }

    meshData.face_counts.resize(n_pts);
    for (int file_id = 0; file_id < n_pts; file_id++) {
        meshData.face_counts[file_id] = (int)mesh->face_counts[original_to_current[file_id]];
    }
}

bool OutputHandler::writeSnapshot(const std::string&     filename,
                                  const MeshCellData&    meshData,
                                  const hydro::primvars* primvar,
                                  int                    n_hydro,
                                  double                 t_sim) {
    std::string fullPath = outputDirectory + filename;

    std::cout << "OUTPUT: Writing snapshot to: " << fullPath << std::endl;

    // create HDF5 file
    hid_t file_id = H5Fcreate(fullPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "OUTPUT: Error! Could not create HDF5 file: " << fullPath << std::endl;
        return false;
    }

    bool success = true;

    // create and write header group
    hid_t header_group = H5Gcreate(file_id, "header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (header_group < 0) {
        std::cerr << "OUTPUT: Error! Could not create header group" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // write header attributes
    hid_t scalar_space = H5Screate(H5S_SCALAR);

    hid_t attr_dim = H5Acreate(header_group, "dimension", H5T_NATIVE_INT, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_dim, H5T_NATIVE_INT, &meshData.header.dimension);
    H5Aclose(attr_dim);

    hid_t attr_extent = H5Acreate(header_group, "extent", H5T_NATIVE_DOUBLE, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_extent, H5T_NATIVE_DOUBLE, &meshData.header.extent);
    H5Aclose(attr_extent);

    hid_t attr_n = H5Acreate(header_group, "n", H5T_NATIVE_INT, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_n, H5T_NATIVE_INT, &meshData.header.n);
    H5Aclose(attr_n);

    hid_t attr_k = H5Acreate(header_group, "k", H5T_NATIVE_INT, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_k, H5T_NATIVE_INT, &meshData.header.k);
    H5Aclose(attr_k);

    hid_t attr_nmax = H5Acreate(header_group, "nmax", H5T_NATIVE_INT, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_nmax, H5T_NATIVE_INT, &meshData.header.nmax);
    H5Aclose(attr_nmax);

    hid_t attr_seed = H5Acreate(header_group, "seed", H5T_NATIVE_INT, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_seed, H5T_NATIVE_INT, &meshData.header.seed);
    H5Aclose(attr_seed);

    hid_t attr_time = H5Acreate(header_group, "time", H5T_NATIVE_DOUBLE, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_time, H5T_NATIVE_DOUBLE, &t_sim);
    H5Aclose(attr_time);

    H5Sclose(scalar_space);
    H5Gclose(header_group);

    // create mesh cells group
    hid_t mesh_group = H5Gcreate(file_id, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (mesh_group < 0) {
        std::cerr << "OUTPUT: Error! Could not create mesh group" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // write seed (generating point) positions
    if (!meshData.seeds.empty() && meshData.seeds_dims.size() == 2) {
        hid_t dataspace = H5Screate_simple(2, meshData.seeds_dims.data(), NULL);
        hid_t dataset_id =
            H5Dcreate(mesh_group, "pos", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id >= 0) {
            H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, meshData.seeds.data());
            H5Dclose(dataset_id);
        }
        H5Sclose(dataspace);
    }

    // write volumes
    if (!meshData.volumes.empty()) {
        hsize_t dims_1d[1]   = {meshData.volumes.size()};
        hid_t   dataspace_1d = H5Screate_simple(1, dims_1d, NULL);
        hid_t   dataset_id =
            H5Dcreate(mesh_group, "volume", H5T_NATIVE_DOUBLE, dataspace_1d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id >= 0) {
            H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, meshData.volumes.data());
            H5Dclose(dataset_id);
        }
        H5Sclose(dataspace_1d);
    }

    // write face_counts (number of faces per cell) (disabled)
    /*
    if (!meshData.face_counts.empty()) {
        hsize_t dims_1d[1]   = {meshData.face_counts.size()};
        hid_t   dataspace_1d = H5Screate_simple(1, dims_1d, NULL);
        hid_t   dataset_id =
            H5Dcreate(mesh_group, "face_counts", H5T_NATIVE_INT, dataspace_1d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id >= 0) {
            H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, meshData.face_counts.data());
            H5Dclose(dataset_id);
        }
        H5Sclose(dataspace_1d);
    }*/

    H5Gclose(mesh_group);

    // create hydro group
    hid_t hydro_group = H5Gcreate(file_id, "hydro", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (hydro_group < 0) {
        std::cerr << "OUTPUT: Warning! Could not create hydro group" << std::endl;
    } else {

        // write rho (density)
        if (primvar && primvar->rho) {
            hsize_t dims_1d[1]   = {(hsize_t)n_hydro};
            hid_t   dataspace_1d = H5Screate_simple(1, dims_1d, NULL);
            hid_t   dataset_id =
                H5Dcreate(hydro_group, "rho", H5T_NATIVE_DOUBLE, dataspace_1d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (dataset_id >= 0) {
                H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, primvar->rho);
                H5Dclose(dataset_id);
            }
            H5Sclose(dataspace_1d);
        }

        // write vel (velocity) - convert from POINT_TYPE to flattened array
        if (primvar && primvar->v) {
            std::vector<double> vel_flat(n_hydro * DIMENSION);
            for (int i = 0; i < n_hydro; i++) {
                vel_flat[i * DIMENSION + 0] = primvar->v[i].x;
                vel_flat[i * DIMENSION + 1] = primvar->v[i].y;
#ifdef dim_3D
                vel_flat[i * DIMENSION + 2] = primvar->v[i].z;
#endif
            }

            hsize_t dims_2d[2] = {(hsize_t)n_hydro, DIMENSION};
            hid_t   dataspace  = H5Screate_simple(2, dims_2d, NULL);
            hid_t   dataset_id =
                H5Dcreate(hydro_group, "vel", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (dataset_id >= 0) {
                H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vel_flat.data());
                H5Dclose(dataset_id);
            }
            H5Sclose(dataspace);
        }

        // write Energy
        if (primvar && primvar->E) {
            hsize_t dims_1d[1]   = {(hsize_t)n_hydro};
            hid_t   dataspace_1d = H5Screate_simple(1, dims_1d, NULL);
            hid_t   dataset_id   = H5Dcreate(
                hydro_group, "energy", H5T_NATIVE_DOUBLE, dataspace_1d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (dataset_id >= 0) {
                H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, primvar->E);
                H5Dclose(dataset_id);
            }
            H5Sclose(dataspace_1d);
        }

        H5Gclose(hydro_group);
    }

    H5Fclose(file_id);
    return success;
}

// prints current step, t, dt and ETA
void print_log(int                                   step,
               std::chrono::steady_clock::time_point wall,
               double                                t_sim,
               double                                dt,
               double                                t_start,
               double                                t_end) {

    const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall).count();
    std::cout << std::endl << "HYDRO: Step " << step << "  t = " << t_sim << "  dt = " << dt
              << "  ETA = " << format_hms((t_sim > t_start) ? elapsed_s * (t_end - t_sim) / (t_sim - t_start) : 0.0)
              << std::endl;
}
