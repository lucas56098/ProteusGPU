#include "input.h"
#include "../global/allvars.h"
#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>

InputHandler::InputHandler(const std::string& filename) : paramFilePath(filename) {}

// helper function to trim whitespace from a string
std::string InputHandler::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// load parameters from parameter file
bool InputHandler::loadParameters() {

    std::ifstream file(paramFilePath);

    // check if file opened successfully
    if (!file.is_open()) {
        std::cerr << "INPUT: Error! Could not open parameter file: " << paramFilePath << std::endl;
        return false;
    }

    // read file line by line
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);

        // skip empty lines and comments
        if (line.empty() || line[0] == '#') { continue; }

        // parse key = value pairs
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key   = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));

            // remove inline comments
            size_t commentPos = value.find('#');
            if (commentPos != std::string::npos) { value = trim(value.substr(0, commentPos)); }

            parameters[key] = value;

#ifdef DEBUG_MODE
            std::cout << "DEBUG: Loaded parameter: " << key << " = " << value << std::endl;
#endif
        }
    }

    file.close();
    std::cout << "INPUT: Loaded " << parameters.size() << " parameters from " << paramFilePath << std::endl;
    return true;
}

// access parameters
std::string InputHandler::getParameter(const std::string& key) const {
    auto it = parameters.find(key);
    if (it != parameters.end()) { return it->second; }
    throw std::runtime_error("Error: Required parameter '" + key + "' not found in parameter file");
}

// get parameter as int
int InputHandler::getParameterInt(const std::string& key) const {
    auto it = parameters.find(key);
    if (it == parameters.end()) {
        throw std::runtime_error("Error: Required parameter '" + key + "' not found in parameter file");
    }
    try {
        return std::stoi(it->second);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error: Could not convert parameter '" + key + "' with value '" + it->second +
                                 "' to int");
    }
}

// get parameter as double
double InputHandler::getParameterDouble(const std::string& key) const {
    auto it = parameters.find(key);
    if (it == parameters.end()) {
        throw std::runtime_error("Error: Required parameter '" + key + "' not found in parameter file");
    }
    try {
        return std::stod(it->second);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error: Could not convert parameter '" + key + "' with value '" + it->second +
                                 "' to double");
    }
}

// get parameter as bool
bool InputHandler::getParameterBool(const std::string& key) const {
    auto it = parameters.find(key);
    if (it == parameters.end()) {
        throw std::runtime_error("Error: Required parameter '" + key + "' not found in parameter file");
    }
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if (value == "true" || value == "1" || value == "yes" || value == "on") {
        return true;
    } else if (value == "false" || value == "0" || value == "no" || value == "off") {
        return false;
    }
    throw std::runtime_error("Error: Could not convert parameter '" + key + "' with value '" + it->second +
                             "' to bool (expected: true/false/1/0/yes/no/on/off)");
}

// check if a parameter exists
bool InputHandler::hasParameter(const std::string& key) const {
    return parameters.find(key) != parameters.end();
}

#ifdef USE_HDF5

// find the latest snapshot_N.hdf5 in a directory, return N (or -1 if none found)
int InputHandler::findLatestSnapshot(const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (!d) return -1;
    int            max_num = -1;
    struct dirent* entry;
    while ((entry = readdir(d)) != NULL) {
        std::string name(entry->d_name);
        if (name.size() > 14 && name.substr(0, 9) == "snapshot_" && name.substr(name.size() - 5) == ".hdf5") {
            std::string num_str = name.substr(9, name.size() - 14);
            try {
                int num = std::stoi(num_str);
                if (num > max_num) max_num = num;
            } catch (...) {}
        }
    }
    closedir(d);
    return max_num;
}

// opens IC.hdf5 file and reads initial conditions into ICData struct
bool InputHandler::readICFile(const std::string& filename, ICData& icData) {

    // check that file exists
    std::ifstream f(filename);
    if (!f.good()) {
        std::cerr << "INPUT: Error! IC file [" << filename << "] does not exist!" << std::endl;
        return false;
    }

    // open the file
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "INPUT: Error! Could not open IC file: " << filename << std::endl;
        return false;
    }

    // read header attributes
    hid_t header_group = H5Gopen(file_id, "header", H5P_DEFAULT);
    if (header_group < 0) {
        std::cerr << "INPUT: Error! Could not open header group" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // read dimension attribute
    hid_t attr_dim = H5Aopen(header_group, "dimension", H5P_DEFAULT);
    if (attr_dim >= 0) {
        H5Aread(attr_dim, H5T_NATIVE_INT, &icData.header.dimension);
        H5Aclose(attr_dim);
    } else {
        std::cerr << "INPUT: Error! Could not read dimension attribute from IC file" << std::endl;
        H5Gclose(header_group);
        H5Fclose(file_id);
        return false;
    }

// check that IC file dimension matches compiled code dimension
#ifdef dim_2D
    if (icData.header.dimension != 2)
#else
    if (icData.header.dimension != 3)
#endif
    {
        std::cerr << "INPUT: Error! IC file dimension mismatch!" << std::endl;
        std::cerr << "  IC file dimension: " << icData.header.dimension << "D" << std::endl;
        std::cerr << "  Compiled code dimension: " << DIMENSION << "D" << std::endl;
        std::cerr << "  Please recompile with correct dimension in Config.sh or use a different IC file." << std::endl;
        H5Gclose(header_group);
        H5Fclose(file_id);
        return false;
    }

    H5Gclose(header_group);

    // read seedpos dataset
    hid_t dataset_id = H5Dopen(file_id, "seedpos", H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "INPUT: Error! Could not open seedpos dataset" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // get dataspace and dimensions
    hid_t dataspace_id = H5Dget_space(dataset_id);
    int   rank         = H5Sget_simple_extent_ndims(dataspace_id);

    if (rank != 2) {
        std::cerr << "INPUT: Error! seedpos dataset must be of shape N x DIM" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    icData.seedpos_dims.resize(2);
    H5Sget_simple_extent_dims(dataspace_id, icData.seedpos_dims.data(), NULL);

    // read the data
    hsize_t totalElements = icData.seedpos_dims[0] * icData.seedpos_dims[1];
    icData.seedpos.resize(totalElements);
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, icData.seedpos.data());

    if (status < 0) {
        std::cerr << "INPUT: Error! Could not read seedpos data" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);

    // read rho dataset
    dataset_id = H5Dopen(file_id, "rho", H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "INPUT: Error! Could not open rho dataset" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    dataspace_id = H5Dget_space(dataset_id);
    hsize_t rho_dims[1];
    H5Sget_simple_extent_dims(dataspace_id, rho_dims, NULL);
    icData.rho.resize(rho_dims[0]);

    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, icData.rho.data());

    if (status < 0) {
        std::cerr << "INPUT: Error! Could not read rho data" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);

    // read vel dataset
    dataset_id = H5Dopen(file_id, "vel", H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "INPUT: Error! Could not open vel dataset" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    dataspace_id = H5Dget_space(dataset_id);
    rank         = H5Sget_simple_extent_ndims(dataspace_id);

    if (rank != 2) {
        std::cerr << "INPUT: Error! vel dataset must be of shape N x DIM" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    hsize_t vel_dims[2];
    H5Sget_simple_extent_dims(dataspace_id, vel_dims, NULL);
    totalElements = vel_dims[0] * vel_dims[1];
    icData.vel.resize(totalElements);

    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, icData.vel.data());

    if (status < 0) {
        std::cerr << "INPUT: Error! Could not read vel data" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);

    // read Energy dataset
    dataset_id = H5Dopen(file_id, "Energy", H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "INPUT: Error! Could not open Energy dataset" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    dataspace_id = H5Dget_space(dataset_id);
    hsize_t Energy_dims[1];
    H5Sget_simple_extent_dims(dataspace_id, Energy_dims, NULL);
    icData.Energy.resize(Energy_dims[0]);

    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, icData.Energy.data());

    if (status < 0) {
        std::cerr << "INPUT: Error! Could not read Energy data" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    std::cout << "INPUT: IC file loaded successfully!" << std::endl;
    return true;
}

// read a snapshot file into ICData for restart (seeds from cells/seeds, hydro from hydro/)
bool InputHandler::readSnapshotFile(const std::string& filename, ICData& icData, double& t_sim) {

    std::ifstream f(filename);
    if (!f.good()) {
        std::cerr << "INPUT: Error! Snapshot file [" << filename << "] does not exist!" << std::endl;
        return false;
    }

    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "INPUT: Error! Could not open snapshot file: " << filename << std::endl;
        return false;
    }

    // read header/time
    hid_t header_group = H5Gopen(file_id, "header", H5P_DEFAULT);
    if (header_group < 0) {
        std::cerr << "INPUT: Error! Could not open header group in snapshot" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    hid_t attr_dim = H5Aopen(header_group, "dimension", H5P_DEFAULT);
    if (attr_dim >= 0) {
        H5Aread(attr_dim, H5T_NATIVE_INT, &icData.header.dimension);
        H5Aclose(attr_dim);
    }

#ifdef dim_2D
    if (icData.header.dimension != 2)
#else
    if (icData.header.dimension != 3)
#endif
    {
        std::cerr << "INPUT: Error! Snapshot dimension mismatch! Snapshot: " << icData.header.dimension
                  << "D, compiled: " << DIMENSION << "D" << std::endl;
        H5Gclose(header_group);
        H5Fclose(file_id);
        return false;
    }

    hid_t attr_time = H5Aopen(header_group, "time", H5P_DEFAULT);
    if (attr_time >= 0) {
        H5Aread(attr_time, H5T_NATIVE_DOUBLE, &t_sim);
        H5Aclose(attr_time);
    } else {
        std::cerr << "INPUT: Error! Could not read time attribute from snapshot" << std::endl;
        H5Gclose(header_group);
        H5Fclose(file_id);
        return false;
    }
    H5Gclose(header_group);

    // read cells/seeds -> icData.seedpos
    hid_t cells_group = H5Gopen(file_id, "cells", H5P_DEFAULT);
    if (cells_group < 0) {
        std::cerr << "INPUT: Error! Could not open cells group in snapshot" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    hid_t dataset_id = H5Dopen(cells_group, "seeds", H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "INPUT: Error! Could not open seeds dataset in snapshot" << std::endl;
        H5Gclose(cells_group);
        H5Fclose(file_id);
        return false;
    }

    hid_t dataspace_id = H5Dget_space(dataset_id);
    icData.seedpos_dims.resize(2);
    H5Sget_simple_extent_dims(dataspace_id, icData.seedpos_dims.data(), NULL);
    hsize_t totalElements = icData.seedpos_dims[0] * icData.seedpos_dims[1];
    icData.seedpos.resize(totalElements);
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, icData.seedpos.data());
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Gclose(cells_group);

    // read hydro/rho
    hid_t hydro_group = H5Gopen(file_id, "hydro", H5P_DEFAULT);
    if (hydro_group < 0) {
        std::cerr << "INPUT: Error! Could not open hydro group in snapshot" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    hsize_t n = icData.seedpos_dims[0];

    dataset_id = H5Dopen(hydro_group, "rho", H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "INPUT: Error! Could not open rho dataset in snapshot" << std::endl;
        H5Gclose(hydro_group);
        H5Fclose(file_id);
        return false;
    }
    icData.rho.resize(n);
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, icData.rho.data());
    H5Dclose(dataset_id);

    // read hydro/vel
    dataset_id = H5Dopen(hydro_group, "vel", H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "INPUT: Error! Could not open vel dataset in snapshot" << std::endl;
        H5Gclose(hydro_group);
        H5Fclose(file_id);
        return false;
    }
    icData.vel.resize(n * DIMENSION);
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, icData.vel.data());
    H5Dclose(dataset_id);

    // read hydro/Energy
    dataset_id = H5Dopen(hydro_group, "Energy", H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "INPUT: Error! Could not open Energy dataset in snapshot" << std::endl;
        H5Gclose(hydro_group);
        H5Fclose(file_id);
        return false;
    }
    icData.Energy.resize(n);
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, icData.Energy.data());
    H5Dclose(dataset_id);

    H5Gclose(hydro_group);
    H5Fclose(file_id);

    std::cout << "INPUT: Snapshot loaded successfully! (" << n << " cells, t = " << t_sim << ")" << std::endl;
    return true;
}
#endif
