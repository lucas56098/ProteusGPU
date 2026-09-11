#include "../global/allvars.h"
#include "h5.h"
#include "input.h"
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef USE_MPI
#include <mpi.h>
#endif

// ============================================================
// read from parameter file
// ============================================================

// load parameters from file
bool InputHandler::loadParameters(const std::string& filename) {

    paramFilePath = filename;
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

bool InputHandler::hasParameter(const std::string& key) const {
    return parameters.find(key) != parameters.end();
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
    h5::File file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    if (!file.valid()) {
        std::cerr << "INPUT: Error! Could not open IC file: " << filename << std::endl;
        return false;
    }

    // read header attributes
    {
        h5::Group header_group(H5Gopen(file, "header", H5P_DEFAULT));
        if (!h5::read_attr(header_group, "dimension", icData.header.dimension)) { return false; }
    }

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
        return false;
    }

    // read mesh/pos and hydro/{rho,vel,energy}
    {
        h5::Group mesh_group(H5Gopen(file, "mesh", H5P_DEFAULT));
        h5::Group hydro_group(H5Gopen(file, "hydro", H5P_DEFAULT));
        if (!h5::read_dataset_2d(mesh_group, "pos", icData.pos, &icData.header.n_seeds) ||
            !h5::read_dataset_1d(hydro_group, "rho", icData.rho) ||
            !h5::read_dataset_2d(hydro_group, "vel", icData.vel) ||
            !h5::read_dataset_1d(hydro_group, "energy", icData.energy)) {
            return false;
        }
    }

    logging::root() << "INPUT: IC file " << filename << " loaded successfully!" << std::endl;

    const int n_total = (int)icData.header.n_seeds;

    // set sequential global IDs in input order
    icData.global_id.resize(n_total);
    for (int i = 0; i < n_total; i++)
        icData.global_id[i] = (uint64_t)i;

    return true;
}

// peek IC file header + global particle count without reading the bulk arrays.
// Opens serially on every rank (independent, no MPIIO setup) since it's a few bytes.
bool InputHandler::readICHeader(const std::string& filename, ICHeader& header, uint64_t& n_total) {

    std::ifstream f(filename);
    if (!f.good()) {
        std::cerr << "INPUT: Error! IC file [" << filename << "] does not exist!" << std::endl;
        return false;
    }

    h5::File file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    if (!file.valid()) {
        std::cerr << "INPUT: Error! Could not open IC file: " << filename << std::endl;
        return false;
    }

    // header/dimension
    {
        h5::Group header_group(H5Gopen(file, "header", H5P_DEFAULT));
        if (!h5::read_attr(header_group, "dimension", header.dimension)) { return false; }
    }

    // n_total from "mesh/pos" dataset extent
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

// collective parallel-HDF5 read of rows [row_lo, row_lo + n_local) for this rank.
bool InputHandler::readICChunkParallel(const std::string& filename, ICData& icData, uint64_t row_lo, uint64_t n_local) {

    // collective open via MPIIO
    h5::File file;
    {
        h5::Plist fapl(H5Pcreate(H5P_FILE_ACCESS));
        H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
        file = h5::File(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, fapl));
    }
    if (!file.valid()) {
        std::cerr << "INPUT: Error! Could not open IC file (parallel): " << filename << std::endl;
        return false;
    }

    // dimension check — same gate as the serial reader
    {
        h5::Group header_group(H5Gopen(file, "header", H5P_DEFAULT));
        if (!h5::read_attr(header_group, "dimension", icData.header.dimension)) { return false; }
    }

#ifdef dim_2D
    if (icData.header.dimension != 2)
#else
    if (icData.header.dimension != 3)
#endif
    {
        std::cerr << "INPUT: Error! IC file dimension mismatch!" << std::endl;
        std::cerr << "  IC file dimension: " << icData.header.dimension << "D" << std::endl;
        std::cerr << "  Compiled code dimension: " << DIMENSION << "D" << std::endl;
        return false;
    }

    icData.header.n_seeds = n_local;

    {
        h5::Group mesh_group(H5Gopen(file, "mesh", H5P_DEFAULT));
        h5::Group hydro_group(H5Gopen(file, "hydro", H5P_DEFAULT));
        if (!h5::read_hyperslab_2d(mesh_group, "pos", row_lo, n_local, (hsize_t)DIMENSION, icData.pos) ||
            !h5::read_hyperslab_2d(hydro_group, "vel", row_lo, n_local, (hsize_t)DIMENSION, icData.vel) ||
            !h5::read_hyperslab_1d(hydro_group, "rho", row_lo, n_local, icData.rho) ||
            !h5::read_hyperslab_1d(hydro_group, "energy", row_lo, n_local, icData.energy)) {
            return false;
        }
    }

    // global IDs in input order: row_lo + i
    icData.global_id.resize(n_local);
    for (uint64_t i = 0; i < n_local; i++)
        icData.global_id[i] = (uint64_t)(row_lo + i);

    logging::root() << "INPUT: IC file " << filename << " loaded in parallel (per-rank chunked read)." << std::endl;
    return true;
}

#endif // USE_MPI

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

    h5::File file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    if (!file.valid()) {
        std::cerr << "INPUT: Error! Could not open snapshot file: " << filename << std::endl;
        return false;
    }

    // read header
    {
        h5::Group header_group(H5Gopen(file, "header", H5P_DEFAULT));
        if (!h5::read_attr(header_group, "dimension", icData.header.dimension) ||
            !h5::read_attr(header_group, "time", snap.t_sim) || !h5::read_attr(header_group, "step", snap.step) ||
            !h5::read_attr(header_group, "n_global", snap.n_global) ||
            !h5::read_attr(header_group, "nranks", snap.nranks) || !h5::read_attr(header_group, "rank", snap.rank)) {
            return false;
        }

        // /header/profiler: walk attrs to recover per-rank cumulative seconds. Snapshots
        // written by the legacy text-log code path lack this sub-group; leave map empty.
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

#ifdef dim_2D
    if (icData.header.dimension != 2)
#else
    if (icData.header.dimension != 3)
#endif
    {
        std::cerr << "INPUT: Error! Snapshot dimension mismatch! Snapshot: " << icData.header.dimension
                  << "D, compiled: " << DIMENSION << "D" << std::endl;
        return false;
    }

    // read mesh/pos and hydro/{rho,vel,energy}
    {
        h5::Group mesh_group(H5Gopen(file, "mesh", H5P_DEFAULT));
        h5::Group hydro_group(H5Gopen(file, "hydro", H5P_DEFAULT));
        if (!h5::read_dataset_2d(mesh_group, "pos", icData.pos, &icData.header.n_seeds) ||
            !h5::read_dataset_1d(hydro_group, "rho", icData.rho) ||
            !h5::read_dataset_2d(hydro_group, "vel", icData.vel) ||
            !h5::read_dataset_1d(hydro_group, "energy", icData.energy)) {
            return false;
        }
    }

    logging::root() << "INPUT: Snapshot loaded successfully! (" << icData.header.n_seeds << " cells, t = " << snap.t_sim
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
