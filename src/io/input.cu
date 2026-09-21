// implements InputHandler (input.h)

#include "../global/allvars.h"
#include "../mpi/mpi_compat.h"
#include "h5.h"
#include "input.h"
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef USE_MPI
#include <mpi.h>
#endif

// ==========================================================
// parameters
// ==========================================================

// loads parameters from param.txt
bool InputHandler::load_parameters(const std::string& filename) {

    param_file_path = filename;
    std::ifstream file(param_file_path);

    if (!file.is_open()) {
        std::cerr << "INPUT: Error! Could not open parameter file: " << param_file_path << std::endl;
        return false;
    }

    std::string line;
    int         line_no = 0;
    while (std::getline(file, line)) { // loop through lines
        line_no++;
        line = trim(line);

        if (line.empty() || line[0] == '#') { continue; } // # is a comment

        size_t pos = line.find('='); // find key = value pairs
        if (pos == std::string::npos || trim(line.substr(0, pos)).empty()) {
            std::cerr << "INPUT: Error! Line " << line_no << " of " << param_file_path
                      << " is not 'key = value': " << line << std::endl;
            return false;
        }
        std::string key   = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));

        size_t comment_pos = value.find('#'); // key = value   #comment
        if (comment_pos != std::string::npos) { value = trim(value.substr(0, comment_pos)); }

        // add key + value to parameters; err if key set twice
        if (!parameters.emplace(key, value).second) {
            std::cerr << "INPUT: Error! Parameter '" << key << "' is set twice in " << param_file_path
                      << " (again on line " << line_no << ")" << std::endl;
            return false;
        }
    }

    file.close();
    logging::root() << "INPUT: Loaded " << parameters.size() << " parameters from " << param_file_path << std::endl;
    return true;
}

// returns parameter for given key
std::string InputHandler::get_parameter(const std::string& key) const {
    auto it = parameters.find(key);

    if (it != parameters.end()) {
        read_keys.insert(key); // store key as read
        return it->second;     // return parameter
    }

    proteus_mpi::exit_failure(
        "INPUT: Error! Required parameter '%s' not found in %s\n", key.c_str(), param_file_path.c_str());
}

// checks if a parameter(key) exists
bool InputHandler::has_parameter(const std::string& key) const {
    if (parameters.find(key) == parameters.end()) { return false; }
    read_keys.insert(key); // if it exists store as read
    return true;
}

// returns parameter(key) as double
double InputHandler::get_parameter_double(const std::string& key) const {
    const std::string value = get_parameter(key);

    // ensure that 0,02 or 3abc dont work
    const char* begin = value.c_str();
    char*       end   = nullptr;
    errno             = 0;
    const double v    = std::strtod(begin, &end);
    if (end == begin || *end != '\0' || errno == ERANGE || !std::isfinite(v)) {
        proteus_mpi::exit_failure("INPUT: Error! Parameter '%s' in %s has value '%s', which is not a finite number\n",
                                  key.c_str(),
                                  param_file_path.c_str(),
                                  value.c_str());
    }
    return v;
}

// returns parameter(key) as int
int InputHandler::get_parameter_int(const std::string& key) const {
    const double v = get_parameter_double(key);

    // ensure 1e1 passes but 10.7 does not
    if (v != std::floor(v) || v < (double)INT_MIN || v > (double)INT_MAX) {
        proteus_mpi::exit_failure("INPUT: Error! Parameter '%s' in %s has value '%s', which is not an integer\n",
                                  key.c_str(),
                                  param_file_path.c_str(),
                                  parameters.at(key).c_str());
    }
    return (int)v;
}

// warns if parameters written in param.txt are unused in the code
void InputHandler::warn_unread_parameters() const {
    std::string unread;

    // loop over parameters
    for (const auto& kv : parameters) {
        // if parameter is not read add it to unread
        if (read_keys.count(kv.first) == 0) { unread += (unread.empty() ? "" : ", ") + kv.first; }
    }

    if (!unread.empty()) {
        logging::root() << "INPUT: Warning! Parameters in " << param_file_path << " not read by this run: " << unread
                        << std::endl;
    }
}

// ==========================================================
// IC files
// ==========================================================

// read whole IC file into ic_data (used in non MPI builds only)
bool InputHandler::read_ic_file(const std::string& filename, ICData& ic_data) {

    std::ifstream f(filename);

    // does file exist?
    if (!f.good()) {
        std::cerr << "INPUT: Error! IC file [" << filename << "] does not exist!" << std::endl;
        return false;
    }

    // open it
    h5::File file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    if (!file.valid()) {
        std::cerr << "INPUT: Error! Could not open IC file: " << filename << std::endl;
        return false;
    }

    // read header
    {
        h5::Group header_group(H5Gopen(file, "header", H5P_DEFAULT));
        if (!h5::read_attr(header_group, "dimension", ic_data.header.dimension)) { return false; }
    }

    // IC dimension MUST align with code dimension
#ifdef dim_2D
    if (ic_data.header.dimension != 2)
#else
    if (ic_data.header.dimension != 3)
#endif
    {
        std::cerr << "INPUT: Error! IC file dimension mismatch!" << std::endl;
        std::cerr << "  IC file dimension: " << ic_data.header.dimension << "D" << std::endl;
        std::cerr << "  Compiled code dimension: " << DIMENSION << "D" << std::endl;
        std::cerr << "  Please recompile with correct dimension in Config.sh or use a different IC file." << std::endl;
        return false;
    }

    // read IC data (mesh + hydro)
    {
        h5::Group mesh_group(H5Gopen(file, "mesh", H5P_DEFAULT));
        h5::Group hydro_group(H5Gopen(file, "hydro", H5P_DEFAULT));
        if (!h5::read_dataset_2d(mesh_group, "pos", ic_data.pos, &ic_data.header.n_seeds) ||
            !h5::read_dataset_1d(hydro_group, "rho", ic_data.rho) ||
            !h5::read_dataset_2d(hydro_group, "vel", ic_data.vel) ||
            !h5::read_dataset_1d(hydro_group, "energy", ic_data.energy)) {
            return false;
        }
    }

    logging::root() << "INPUT: IC file " << filename << " loaded successfully!" << std::endl;

    return true;
}

// read ic header only (dimension + total cell count), before any cells are read
bool InputHandler::read_ic_header(const std::string& filename, ICHeader& header, uint64_t& n_total) {

    std::ifstream f(filename);

    // does file exist?
    if (!f.good()) {
        std::cerr << "INPUT: Error! IC file [" << filename << "] does not exist!" << std::endl;
        return false;
    }

    // open it
    h5::File file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    if (!file.valid()) {
        std::cerr << "INPUT: Error! Could not open IC file: " << filename << std::endl;
        return false;
    }

    // read header
    {
        h5::Group header_group(H5Gopen(file, "header", H5P_DEFAULT));
        if (!h5::read_attr(header_group, "dimension", header.dimension)) { return false; }
    }

    // read mesh/pos extent (n_total cells)
    h5::Dataset dset(H5Dopen(file, "mesh/pos", H5P_DEFAULT));
    if (!dset.valid()) {
        std::cerr << "INPUT: Error! Could not open dataset 'mesh/pos' for header peek" << std::endl;
        return false;
    }
    h5::Space space(H5Dget_space(dset));
    if (!h5::check_rank(space, "mesh/pos", 2)) { return false; }
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, NULL);
    n_total = dims[0];

    return true;
}

#ifdef USE_MPI

// each rank reads its own IC rows [row_lo, row_lo + n_local) through MPI-IO
bool InputHandler::read_ic_chunk_parallel(const std::string& filename,
                                          ICData&            ic_data,
                                          uint64_t           row_lo,
                                          uint64_t           n_local) {

    h5::File file;

    // open file
    {
        h5::Plist fapl(H5Pcreate(H5P_FILE_ACCESS));
        H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
        file = h5::File(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, fapl));
    }
    if (!file.valid()) {
        std::cerr << "INPUT: Error! Could not open IC file (parallel): " << filename << std::endl;
        return false;
    }

    // read header
    {
        h5::Group header_group(H5Gopen(file, "header", H5P_DEFAULT));
        if (!h5::read_attr(header_group, "dimension", ic_data.header.dimension)) { return false; }
    }

    // check dimensions agree
#ifdef dim_2D
    if (ic_data.header.dimension != 2)
#else
    if (ic_data.header.dimension != 3)
#endif
    {
        std::cerr << "INPUT: Error! IC file dimension mismatch!" << std::endl;
        std::cerr << "  IC file dimension: " << ic_data.header.dimension << "D" << std::endl;
        std::cerr << "  Compiled code dimension: " << DIMENSION << "D" << std::endl;
        return false;
    }

    // local cell count
    ic_data.header.n_seeds = n_local;

    // read local part of IC (distribution following domain decomp happens later)
    {
        h5::Group mesh_group(H5Gopen(file, "mesh", H5P_DEFAULT));
        h5::Group hydro_group(H5Gopen(file, "hydro", H5P_DEFAULT));
        if (!h5::read_hyperslab_2d(mesh_group, "pos", row_lo, n_local, (hsize_t)DIMENSION, ic_data.pos) ||
            !h5::read_hyperslab_2d(hydro_group, "vel", row_lo, n_local, (hsize_t)DIMENSION, ic_data.vel) ||
            !h5::read_hyperslab_1d(hydro_group, "rho", row_lo, n_local, ic_data.rho) ||
            !h5::read_hyperslab_1d(hydro_group, "energy", row_lo, n_local, ic_data.energy)) {
            return false;
        }
    }

    logging::root() << "INPUT: IC file " << filename << " loaded in parallel (per-rank chunked read)." << std::endl;
    return true;
}

#endif

// ==========================================================
// snapshots
// ==========================================================

// highest N of snapshot_N.hdf5 (1 rank) or snapshot_N.<rank>.hdf5; -1 if none
int InputHandler::find_latest_snapshot(const std::string& dir, int nranks, int rank) {

    DIR* d = opendir(dir.c_str());
    if (!d) return -1;

    const std::string prefix = "snapshot_";
    const std::string suffix = (nranks > 1) ? ("." + std::to_string(rank) + ".hdf5") : std::string(".hdf5");

    int            max_num = -1;
    struct dirent* entry;

    // loop over snapshots to find highest
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

// read a snapshot into ic_data (for restart from snapshot)
bool InputHandler::read_snapshot_file(const std::string& filename, ICData& ic_data, SnapshotHeader& snap) {

    std::ifstream f(filename);

    // does file exist?
    if (!f.good()) {
        std::cerr << "INPUT: Error! Snapshot file [" << filename << "] does not exist!" << std::endl;
        return false;
    }

    // open file
    h5::File file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    if (!file.valid()) {
        std::cerr << "INPUT: Error! Could not open snapshot file: " << filename << std::endl;
        return false;
    }

    // read header
    {
        h5::Group header_group(H5Gopen(file, "header", H5P_DEFAULT));
        if (!h5::read_attr(header_group, "dimension", ic_data.header.dimension) ||
            !h5::read_attr(header_group, "time", snap.t_sim) || !h5::read_attr(header_group, "step", snap.step) ||
            !h5::read_attr(header_group, "n_global", snap.n_global) ||
            !h5::read_attr(header_group, "nranks", snap.nranks) || !h5::read_attr(header_group, "rank", snap.rank)) {
            return false;
        }

        // without it knn::init_once sets ngb grid from current cell count (needed for bitwise reproducible restart)
        if (H5Aexists(header_group, "knn_N_grid") <= 0) {
            std::cerr << "INPUT: Error! Snapshot " << filename
                      << " has no header/knn_N_grid, so it predates reproducible restarts. Resuming from it "
                         "would size the neighbour grid differently and reorder the cells."
                      << std::endl;
            return false;
        }
        if (!h5::read_attr(header_group, "knn_N_grid", ic_data.header.knn_N_grid)) { return false; }

        // optional: cumulative timer values of the old run
        if (H5Lexists(header_group, "profiler", H5P_DEFAULT) > 0) {
            h5::Group    prof_group(H5Gopen(header_group, "profiler", H5P_DEFAULT));
            const herr_t walked = H5Aiterate2(
                prof_group,
                H5_INDEX_NAME,
                H5_ITER_NATIVE,
                NULL,
                [](hid_t loc, const char* name, const H5A_info_t*, void* data) -> herr_t {
                    auto*    map = static_cast<std::unordered_map<std::string, double>*>(data);
                    h5::Attr a(H5Aopen(loc, name, H5P_DEFAULT));
                    double   v = 0.0;
                    if (!a.valid() || H5Aread(a, H5T_NATIVE_DOUBLE, &v) < 0) { return -1; }
                    (*map)[name] = v;
                    return 0;
                },
                &snap.profiler_cum);
            if (walked < 0) {
                std::cerr << "INPUT: Error! Could not read the profiler attributes of " << filename << std::endl;
                return false;
            }
        }
    }

    // check snapshot dimension is same as code dimension
#ifdef dim_2D
    if (ic_data.header.dimension != 2)
#else
    if (ic_data.header.dimension != 3)
#endif
    {
        std::cerr << "INPUT: Error! Snapshot dimension mismatch! Snapshot: " << ic_data.header.dimension
                  << "D, compiled: " << DIMENSION << "D" << std::endl;
        return false;
    }

    // load data from snap
    {
        h5::Group mesh_group(H5Gopen(file, "mesh", H5P_DEFAULT));
        h5::Group hydro_group(H5Gopen(file, "hydro", H5P_DEFAULT));
        if (!h5::read_dataset_2d(mesh_group, "pos", ic_data.pos, &ic_data.header.n_seeds) ||
            !h5::read_dataset_1d(hydro_group, "rho", ic_data.rho) ||
            !h5::read_dataset_2d(hydro_group, "vel", ic_data.vel) ||
            !h5::read_dataset_1d(hydro_group, "energy", ic_data.energy)) {
            return false;
        }

#ifdef MOVING_MESH
        // load v_mesh as its only written in hydro_step; else first dt after restart would use zeros
        if (H5Lexists(mesh_group, "v_mesh", H5P_DEFAULT) <= 0) {
            std::cerr << "INPUT: Error! Snapshot " << filename
                      << " has no mesh/v_mesh, so it predates moving-mesh restart support. "
                         "Continuing from it would take a different first timestep."
                      << std::endl;
            return false;
        }
        if (!h5::read_dataset_2d(mesh_group, "v_mesh", ic_data.v_mesh)) { return false; }
#endif

#ifdef USE_MPI
        // load prev decomp; otherwise decomp_init starts with even split (needed for bitwise reproducible restart)
        if (H5Lexists(file, "decomp", H5P_DEFAULT) <= 0) {
            std::cerr << "INPUT: Error! Snapshot " << filename
                      << " has no /decomp group, so the split tables it ran with are unknown." << std::endl;
            return false;
        }
        h5::Group   decomp_group(H5Gopen(file, "decomp", H5P_DEFAULT));
        const char* axis_name[3] = {"splits_x", "splits_y", "splits_z"};
        for (int a = 0; a < 3; a++) {
            if (!h5::read_dataset_1d(decomp_group, axis_name[a], ic_data.header.decomp_splits[a])) { return false; }
        }
#endif
    }

    logging::root() << "INPUT: Snapshot loaded successfully! (" << ic_data.header.n_seeds
                    << " cells, t = " << snap.t_sim << ")" << std::endl;
    return true;
}

// find whitspaces, \t, \t, \n and rm from string
std::string InputHandler::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}
