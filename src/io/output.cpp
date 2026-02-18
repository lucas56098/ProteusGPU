#include "output.h"
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

// has to be improved once we are at a point that there actually is something to store...

OutputHandler::OutputHandler(const std::string& outputDir) : outputDirectory(outputDir) {
}

bool OutputHandler::initialize() {
    // create output directory if it doesn't exist
    struct stat st;
    if (stat(outputDirectory.c_str(), &st) != 0) {
        if (mkdir(outputDirectory.c_str(), 0755) != 0) {
            std::cerr << "Error: Could not create output directory: " << outputDirectory << std::endl;
            return false;
        }
        std::cout << "Created new output directory: " << outputDirectory << std::endl;
    }
    std::cout << "Output directory: " << outputDirectory << std::endl;

    return true;
}

#ifdef USE_HDF5
bool OutputHandler::writeMeshFile(const std::string& filename, const MeshCellData& meshData) {
    std::string fullPath = outputDirectory + "/" + filename;
    
    std::cout << "Writing mesh to: " << fullPath << std::endl;
    
    // create HDF5 file
    hid_t file_id = H5Fcreate(fullPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error: Could not create HDF5 file: " << fullPath << std::endl;
        return false;
    }

    bool success = true;

    // create and write header group
    hid_t header_group = H5Gcreate(file_id, "header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (header_group < 0) {
        std::cerr << "Error: Could not create header group" << std::endl;
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

    hid_t attr_store = H5Acreate(header_group, "store_edge_coords", H5T_NATIVE_HBOOL, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
    hbool_t store_bool = meshData.header.store_edge_coords ? 1 : 0;
    H5Awrite(attr_store, H5T_NATIVE_HBOOL, &store_bool);
    H5Aclose(attr_store);

    H5Sclose(scalar_space);
    H5Gclose(header_group);

    std::cout << "Header written successfully" << std::endl;

    // create cells group
    hid_t cells_group = H5Gcreate(file_id, "cells", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (cells_group < 0) {
        std::cerr << "Error: Could not create cells group" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // write cell_ids
    if (!meshData.cell_ids.empty()) {
        hsize_t dims_1d[1] = {meshData.cell_ids.size()};
        hid_t dataspace_1d = H5Screate_simple(1, dims_1d, NULL);
        hid_t dataset_id = H5Dcreate(cells_group, "cell_ids", H5T_NATIVE_INT, dataspace_1d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id >= 0) {
            H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, meshData.cell_ids.data());
            H5Dclose(dataset_id);
            std::cout << "  cell_ids: " << meshData.cell_ids.size() << " cells" << std::endl;
        }
        H5Sclose(dataspace_1d);
    }

    // write seeds
    if (!meshData.seeds.empty() && meshData.seeds_dims.size() == 2) {
        hid_t dataspace = H5Screate_simple(2, meshData.seeds_dims.data(), NULL);
        hid_t dataset_id = H5Dcreate(cells_group, "seeds", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id >= 0) {
            H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, meshData.seeds.data());
            H5Dclose(dataset_id);
            std::cout << "  seeds: " << meshData.seeds_dims[0] << " x " << meshData.seeds_dims[1] << std::endl;
        }
        H5Sclose(dataspace);
    }

    // write volumes
    if (!meshData.volumes.empty()) {
        hsize_t dims_1d[1] = {meshData.volumes.size()};
        hid_t dataspace_1d = H5Screate_simple(1, dims_1d, NULL);
        hid_t dataset_id = H5Dcreate(cells_group, "volumes", H5T_NATIVE_DOUBLE, dataspace_1d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id >= 0) {
            H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, meshData.volumes.data());
            H5Dclose(dataset_id);
            std::cout << "  volumes: " << meshData.volumes.size() << " volumes" << std::endl;
        }
        H5Sclose(dataspace_1d);
    }

    // create faces subgroup
    hid_t faces_group = H5Gcreate(cells_group, "faces", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (faces_group < 0) {
        std::cerr << "Error: Could not create faces group" << std::endl;
        H5Gclose(cells_group);
        H5Fclose(file_id);
        return false;
    }

    // write neighbor_cell
    if (!meshData.faces.neighbor_cell.empty()) {
        hsize_t dims_1d[1] = {meshData.faces.neighbor_cell.size()};
        hid_t dataspace_1d = H5Screate_simple(1, dims_1d, NULL);
        hid_t dataset_id = H5Dcreate(faces_group, "neighbor_cell", H5T_NATIVE_INT, dataspace_1d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id >= 0) {
            H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, meshData.faces.neighbor_cell.data());
            H5Dclose(dataset_id);
            std::cout << "  neighbor_cell: " << meshData.faces.neighbor_cell.size() << " faces" << std::endl;
        }
        H5Sclose(dataspace_1d);
    }

    // write normal
    if (!meshData.faces.normal.empty() && meshData.faces.normal_dims.size() == 2) {
        hid_t dataspace = H5Screate_simple(2, meshData.faces.normal_dims.data(), NULL);
        hid_t dataset_id = H5Dcreate(faces_group, "normal", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id >= 0) {
            H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, meshData.faces.normal.data());
            H5Dclose(dataset_id);
            std::cout << "  normal: " << meshData.faces.normal_dims[0] << " x " << meshData.faces.normal_dims[1] << std::endl;
        }
        H5Sclose(dataspace);
    }

    // write area
    if (!meshData.faces.area.empty()) {
        hsize_t dims_1d[1] = {meshData.faces.area.size()};
        hid_t dataspace_1d = H5Screate_simple(1, dims_1d, NULL);
        hid_t dataset_id = H5Dcreate(faces_group, "area", H5T_NATIVE_DOUBLE, dataspace_1d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id >= 0) {
            H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, meshData.faces.area.data());
            H5Dclose(dataset_id);
            std::cout << "  area: " << meshData.faces.area.size() << " areas" << std::endl;
        }
        H5Sclose(dataspace_1d);
    }

    // write edge_coords if storing them
    if (meshData.header.store_edge_coords && !meshData.faces.edge_coords.empty() && meshData.faces.edge_coords_dims.size() == 2) {
        hid_t dataspace = H5Screate_simple(2, meshData.faces.edge_coords_dims.data(), NULL);
        hid_t dataset_id = H5Dcreate(faces_group, "edge_coords", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id >= 0) {
            H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, meshData.faces.edge_coords.data());
            H5Dclose(dataset_id);
            std::cout << "  edge_coords: " << meshData.faces.edge_coords_dims[0] << " x " << meshData.faces.edge_coords_dims[1] << std::endl;
        }
        H5Sclose(dataspace);
    }

    // write edge_coords_offsets if storing edge coords
    if (meshData.header.store_edge_coords && !meshData.faces.edge_coords_offsets.empty()) {
        hsize_t dims_1d[1] = {meshData.faces.edge_coords_offsets.size()};
        hid_t dataspace_1d = H5Screate_simple(1, dims_1d, NULL);
        hid_t dataset_id = H5Dcreate(faces_group, "edge_coords_offsets", H5T_NATIVE_INT, dataspace_1d, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id >= 0) {
            H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, meshData.faces.edge_coords_offsets.data());
            H5Dclose(dataset_id);
            std::cout << "  edge_coords_offsets: " << meshData.faces.edge_coords_offsets.size() << " offsets" << std::endl;
        }
        H5Sclose(dataspace_1d);
    }

    H5Gclose(faces_group);
    H5Gclose(cells_group);
    H5Fclose(file_id);

    std::cout << "Mesh file written successfully to: " << fullPath << std::endl;
    return success;
}
#endif

