#include "input.h"
#include "../global/allvars.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

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
        std::cerr << "Error: Could not open parameter file: " << paramFilePath << std::endl;
        return false;
    }

    // read file line by line
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        
        // skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // parse key = value pairs
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));
            
            // remove inline comments
            size_t commentPos = value.find('#');
            if (commentPos != std::string::npos) {
                value = trim(value.substr(0, commentPos));
            }
            
            parameters[key] = value;
            
#ifdef DEBUG_MODE
            std::cout << "Loaded parameter: " << key << " = " << value << std::endl;
#endif
        }
    }

    file.close();
    std::cout << "Loaded " << parameters.size() << " parameters from " << paramFilePath << std::endl;
    return true;
}

// access parameters
std::string InputHandler::getParameter(const std::string& key) const {
    auto it = parameters.find(key);
    if (it != parameters.end()) {
        return it->second;
    }
    throw std::runtime_error("Error: Required parameter '" + key + "' not found in parameter file");
}

int InputHandler::getParameterInt(const std::string& key) const {
    auto it = parameters.find(key);
    if (it == parameters.end()) {
        throw std::runtime_error("Error: Required parameter '" + key + "' not found in parameter file");
    }
    try {
        return std::stoi(it->second);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error: Could not convert parameter '" + key + "' with value '" + it->second + "' to int");
    }
}

double InputHandler::getParameterDouble(const std::string& key) const {
    auto it = parameters.find(key);
    if (it == parameters.end()) {
        throw std::runtime_error("Error: Required parameter '" + key + "' not found in parameter file");
    }
    try {
        return std::stod(it->second);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error: Could not convert parameter '" + key + "' with value '" + it->second + "' to double");
    }
}

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
    throw std::runtime_error("Error: Could not convert parameter '" + key + "' with value '" + it->second + "' to bool (expected: true/false/1/0/yes/no/on/off)");
}

#ifdef USE_HDF5
// opens IC.hdf5 file and reads initial conditions into ICData struct
// will get extended if we read more than dimension, extend and seedpoints...
bool InputHandler::readICFile(const std::string& filename, ICData& icData) {
   
    // check that file exists
    std::ifstream f(filename);
    if (!f.good()) {
	std::cerr << "Error: IC file [" << filename << "] does not exist!" << std::endl;
        return false;
    }

    // open the file
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error: Could not open IC file: " << filename << std::endl;
        return false;
    }

    // read header attributes
    hid_t header_group = H5Gopen(file_id, "header", H5P_DEFAULT);
    if (header_group < 0) {
        std::cerr << "Error: Could not open header group" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // read dimension attribute
    hid_t attr_dim = H5Aopen(header_group, "dimension", H5P_DEFAULT);
    if (attr_dim >= 0) {
        H5Aread(attr_dim, H5T_NATIVE_INT, &icData.header.dimension);
        H5Aclose(attr_dim);
    } else {
        std::cerr << "Error: Could not read dimension attribute from IC file" << std::endl;
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
        std::cerr << "Error: IC file dimension mismatch!" << std::endl;
        std::cerr << "  IC file dimension: " << icData.header.dimension << "D" << std::endl;
        std::cerr << "  Compiled code dimension: " << DIMENSION << "D" << std::endl;
        std::cerr << "  Please recompile with correct dimension in Config.sh or use a different IC file." << std::endl;
        H5Gclose(header_group);
        H5Fclose(file_id);
        return false;
    }

    // read extent attribute
    hid_t attr_extent = H5Aopen(header_group, "extent", H5P_DEFAULT);
    if (attr_extent >= 0) {
        H5Aread(attr_extent, H5T_NATIVE_DOUBLE, &icData.header.extent);
        H5Aclose(attr_extent);
    } else {
        std::cerr << "Warning: Could not read extent attribute" << std::endl;
        icData.header.extent = 1000.0;  // default
    }
    H5Gclose(header_group);

    // read seedpos dataset
    hid_t dataset_id = H5Dopen(file_id, "seedpos", H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error: Could not open seedpos dataset" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // get dataspace and dimensions
    hid_t dataspace_id = H5Dget_space(dataset_id);
    int rank = H5Sget_simple_extent_ndims(dataspace_id);
    
    if (rank != 2) {
        std::cerr << "Error: seedpos dataset must be of shape N x DIM" << std::endl;
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
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, icData.seedpos.data());

    if (status < 0) {
        std::cerr << "Error: Could not read seedpos data" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    
    std::cout << "IC file loaded successfully!" << std::endl;
    return true;
}
#endif
