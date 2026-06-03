#include "../global/allvars.h"
#include "input.h"
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>

static void read_attr_int(hid_t group, const char* name, int& out);
static void read_attr_double(hid_t group, const char* name, double& out);
static bool read_dataset_1d(hid_t parent, const char* name, std::vector<double>& out);
static bool read_dataset_2d(hid_t parent, const char* name, std::vector<double>& out, hsize_t* out_dims = nullptr);

InputHandler::InputHandler(const std::string& filename) : paramFilePath(filename) {}

// ============================================================
// read from parameter file
// ============================================================

// load parameters from file
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
        }
    }

    file.close();
    logging::root() << "INPUT: Loaded " << parameters.size() << " parameters from " << paramFilePath << std::endl;
    return true;
}

// access parameter
std::string InputHandler::getParameter(const std::string& key) const {
    auto it = parameters.find(key);
    if (it != parameters.end()) { return it->second; }
    throw std::runtime_error("Error: Required parameter '" + key + "' not found in parameter file");
}

// access parameter converted to double
double InputHandler::getParameterDouble(const std::string& key) const {
    std::string value = getParameter(key);
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Error: Could not convert parameter '" + key + "' with value '" + value +
                                 "' to double");
    }
}

// ============================================================
// load ic
// ============================================================

// read IC file into icData
bool InputHandler::readICFile(const std::string& filename, ICData& icData) {

    // check file exists
    std::ifstream f(filename);
    if (!f.good()) {
        std::cerr << "INPUT: Error! IC file [" << filename << "] does not exist!" << std::endl;
        return false;
    }

    // open file
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "INPUT: Error! Could not open IC file: " << filename << std::endl;
        return false;
    }

    // read header attributes
    hid_t header_group = H5Gopen(file_id, "header", H5P_DEFAULT);
    read_attr_int(header_group, "dimension", icData.header.dimension);
    H5Gclose(header_group);

    // check that IC dimension matches code dimension
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
        H5Fclose(file_id);
        return false;
    }

    // read pos/rho/vel/energy
    icData.pos_dims.resize(2);
    if (!read_dataset_2d(file_id, "pos", icData.pos, icData.pos_dims.data()) ||
        !read_dataset_1d(file_id, "rho", icData.rho) || !read_dataset_2d(file_id, "vel", icData.vel) ||
        !read_dataset_1d(file_id, "energy", icData.energy)) {
        H5Fclose(file_id);
        return false;
    }

    // close file
    H5Fclose(file_id);
    logging::root() << "INPUT: IC file " << filename << " loaded successfully!" << std::endl;
    return true;
}

// ============================================================
// load snapshot
// ============================================================

// find latest snapshot N in directory
int InputHandler::findLatestSnapshot(const std::string& dir, int nranks, int rank) {
    DIR* d = opendir(dir.c_str());
    if (!d) return -1;

    // format: snapshot_*.hdf5 or snapshot_*.rank.hdf5
    const std::string prefix = "snapshot_";
    const std::string suffix = (nranks > 1) ? ("." + std::to_string(rank) + ".hdf5") : std::string(".hdf5");

    int            max_num = -1;
    struct dirent* entry;
    while ((entry = readdir(d)) != NULL) {
        std::string name(entry->d_name);
        if (name.size() <= prefix.size() + suffix.size()) continue;
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;

        std::string num_str = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        try {
            size_t consumed = 0;
            int    num      = std::stoi(num_str, &consumed);
            if (consumed == num_str.size() && num > max_num) max_num = num;
        } catch (...) {}
    }
    closedir(d);
    return max_num;
}

// read snapshot into icData for restart
bool InputHandler::readSnapshotFile(const std::string& filename, ICData& icData, SnapshotHeader& snap) {

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

    // read header
    hid_t header_group = H5Gopen(file_id, "header", H5P_DEFAULT);
    read_attr_int(header_group, "dimension", icData.header.dimension);
    read_attr_double(header_group, "time", snap.t_sim);
    read_attr_int(header_group, "step", snap.step);
    read_attr_int(header_group, "n_global", snap.n_global);
    read_attr_int(header_group, "nranks", snap.nranks);
    read_attr_int(header_group, "rank", snap.rank);
    H5Gclose(header_group);

#ifdef dim_2D
    if (icData.header.dimension != 2)
#else
    if (icData.header.dimension != 3)
#endif
    {
        std::cerr << "INPUT: Error! Snapshot dimension mismatch! Snapshot: " << icData.header.dimension
                  << "D, compiled: " << DIMENSION << "D" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // read mesh/pos and hydro/{rho,vel,energy}
    hid_t mesh_group  = H5Gopen(file_id, "mesh", H5P_DEFAULT);
    hid_t hydro_group = H5Gopen(file_id, "hydro", H5P_DEFAULT);
    icData.pos_dims.resize(2);
    if (!read_dataset_2d(mesh_group, "pos", icData.pos, icData.pos_dims.data()) ||
        !read_dataset_1d(hydro_group, "rho", icData.rho) || !read_dataset_2d(hydro_group, "vel", icData.vel) ||
        !read_dataset_1d(hydro_group, "energy", icData.energy)) {
        H5Gclose(mesh_group);
        H5Gclose(hydro_group);
        H5Fclose(file_id);
        return false;
    }
    H5Gclose(mesh_group);
    H5Gclose(hydro_group);
    H5Fclose(file_id);

    logging::root() << "INPUT: Snapshot loaded successfully! (" << icData.pos_dims[0] << " cells, t = " << snap.t_sim
                    << ")" << std::endl;
    return true;
}

// ============================================================
// helpers
// ============================================================

// trim whitespace from string
std::string InputHandler::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// HDF5 read operations
static void read_attr_int(hid_t group, const char* name, int& out) {
    hid_t a = H5Aopen(group, name, H5P_DEFAULT);
    H5Aread(a, H5T_NATIVE_INT, &out);
    H5Aclose(a);
}

static void read_attr_double(hid_t group, const char* name, double& out) {
    hid_t a = H5Aopen(group, name, H5P_DEFAULT);
    H5Aread(a, H5T_NATIVE_DOUBLE, &out);
    H5Aclose(a);
}

static bool read_dataset_1d(hid_t parent, const char* name, std::vector<double>& out) {
    hid_t dset = H5Dopen(parent, name, H5P_DEFAULT);
    if (dset < 0) {
        std::cerr << "INPUT: Error! Could not open dataset '" << name << "'" << std::endl;
        return false;
    }
    hid_t   space = H5Dget_space(dset);
    hsize_t dim;
    H5Sget_simple_extent_dims(space, &dim, NULL);
    out.resize(dim);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    H5Sclose(space);
    H5Dclose(dset);
    return true;
}

static bool read_dataset_2d(hid_t parent, const char* name, std::vector<double>& out, hsize_t* out_dims) {
    hid_t dset = H5Dopen(parent, name, H5P_DEFAULT);
    if (dset < 0) {
        std::cerr << "INPUT: Error! Could not open dataset '" << name << "'" << std::endl;
        return false;
    }
    hid_t   space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, NULL);
    out.resize(dims[0] * dims[1]);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    H5Sclose(space);
    H5Dclose(dset);
    if (out_dims) {
        out_dims[0] = dims[0];
        out_dims[1] = dims[1];
    }
    return true;
}
