// implements the initial conditions and the snapshots (io.h)

#include "io/io.h"

#include "io/h5.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace io {

    // a vector field is written as rows of config::dim doubles, straight from the array of Vec
    static_assert(sizeof(Vec) == config::dim * sizeof(double), "a Vec is config::dim doubles in a row");

    namespace {

        // a dataset of the cells, with its numbers per row
        struct Field {
            const char* name;
            int         columns;
        };

        hid_t open_file_of_cells(const char* path, const char* what);
        template <typename Number> Number header_number(hid_t file, const char* path, const char* name);
        long rows_of_every_field(hid_t file, const char* path, const std::vector<Field>& fields);
        Rows rows_of_this_rank(long n_all);
        void put_seeds_and_gas_into_cells(hid_t file, const char* path, const Rows& rows, cells::Cells& cells);
        std::string snapshot_path(const std::string& directory, int number);
        Rows rows_after_lower_ranks(long n);
        void write_mesh_geometry(hid_t                        mesh_group,
                                 const mesh::MeshPointers&    mesh,
                                 const ghosts::GhostPointers& ghosts,
                                 const Rows&                  cell_rows);
        void write_steps(hid_t                          hydro,
                         const cells::Cells&            cells,
                         const timesteps::StepPointers& steps,
                         int                            step,
                         const Rows&                    rows);
        std::vector<long long> first_row_of_every_rank(long n);

    } // namespace

    // the energy of the file becomes the pressure; a run starts with the mesh at rest
    void read_initial_conditions(const char* path, cells::Cells& cells) {
        const File file(open_file_of_cells(path, "initial conditions"));
        const long n_all = rows_of_every_field(
            file, path, {{"mesh/pos", config::dim}, {"hydro/rho", 1}, {"hydro/vel", config::dim}, {"hydro/energy", 1}});
        const Rows rows = rows_of_this_rank(n_all);
        put_seeds_and_gas_into_cells(file, path, rows, cells);

        const std::vector<double> energy = read_rows(file, path, "hydro/energy", rows, 1);
        for (long c = 0; c < rows.n; c++) {
            cells.id[c]       = rows.first + c;
            cells.pressure[c] = gas::pressure_of(cells.density[c], cells.velocity[c], energy[c]);
            if constexpr (config::moving_mesh) cells.mesh_velocity[c] = Vec{};
        }
        logging::print("INPUT: IC file %s loaded successfully!", path);
    }

    // the ranks agree whether the first snapshot is there before any rank writes one
    void prepare_output_directory(const std::string& directory) {
        const bool created = mkdir(directory.c_str(), 0755) == 0;
        if (!created && errno != EEXIST) {
            logging::stop_run("OUTPUT: cannot make the output directory %s", directory.c_str());
        }
        if (created) logging::print("OUTPUT: Created new output directory: %s", directory.c_str());
        struct stat       status;
        const std::string first = snapshot_path(directory, 0);
        if (mpi::on_any_rank(stat(first.c_str(), &status) == 0)) {
            logging::stop_run("OUTPUT: %s is there already; a run does not write over another one", first.c_str());
        }
        logging::print("OUTPUT: directory: %s", directory.c_str());
    }

    int last_snapshot_number(const std::string& directory) {
        int  last   = -1;
        DIR* folder = opendir(directory.c_str());
        if (folder != nullptr) {
            while (const dirent* entry = readdir(folder)) {
                int number = -1;
                if (std::sscanf(entry->d_name, "snapshot_%d", &number) == 1 && number >= 0 &&
                    snapshot_path("", number) == entry->d_name) {
                    last = std::max(last, number);
                }
            }
            closedir(folder);
        }
        return mpi::max_over_ranks(last);
    }

    SnapshotHeader read_snapshot(const std::string& directory, int number, cells::Cells& cells) {
        logging::print("RESTART: Loading snapshot snapshot_%d.hdf5 from %s", number, directory.c_str());
        const std::string  path = snapshot_path(directory, number);
        const File         file(open_file_of_cells(path.c_str(), "snapshot"));
        std::vector<Field> fields{{"mesh/pos", config::dim},
                                  {"mesh/id", 1},
                                  {"hydro/rho", 1},
                                  {"hydro/vel", config::dim},
                                  {"hydro/pressure", 1}};
        if constexpr (config::moving_mesh) fields.push_back({"mesh/v_mesh", config::dim});
        const long n_all = rows_of_every_field(file, path.c_str(), fields);
        const Rows rows  = rows_of_this_rank(n_all);
        put_seeds_and_gas_into_cells(file, path.c_str(), rows, cells);

        const std::vector<int64_t> id       = read_rows<int64_t>(file, path.c_str(), "mesh/id", rows, 1);
        const std::vector<double>  pressure = read_rows(file, path.c_str(), "hydro/pressure", rows, 1);
        for (long c = 0; c < rows.n; c++) {
            cells.id[c]       = id[c];
            cells.pressure[c] = pressure[c];
        }
        if constexpr (config::moving_mesh) {
            const std::vector<double> mesh_velocity = read_rows(file, path.c_str(), "mesh/v_mesh", rows, config::dim);
            for (long c = 0; c < rows.n; c++) {
                for (int d = 0; d < config::dim; d++)
                    cells.mesh_velocity[c][d] = mesh_velocity[config::dim * c + d];
            }
        }

        const SnapshotHeader header{header_number<double>(file, path.c_str(), "time"),
                                    header_number<int>(file, path.c_str(), "step")};
        logging::print("INPUT: Snapshot loaded successfully! (%ld cells, t = %g)", n_all, header.time);
        return header;
    }

    // the header, ids, seeds and gas, energy as total energy per volume; with OUTPUT_DEBUG also the mesh geometry
    void write_snapshot(const std::string&             directory,
                        int                            number,
                        double                         time,
                        int                            step,
                        const astro::Units&            units,
                        const cells::Cells&            cells,
                        const mesh::MeshPointers&      mesh,
                        const ghosts::GhostPointers&   ghosts,
                        const timesteps::StepPointers& steps) {
        const int         n_cells = cells.n_cells;
        const Rows        rows    = rows_after_lower_ranks(n_cells);
        const std::string path    = snapshot_path(directory, number);
        File              file(create_by_all_ranks(path));
        if (!file.is_open()) logging::stop_run("OUTPUT: cannot write the snapshot %s", path.c_str());

        {
            const Group header(H5Gcreate(file, "header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            write_attribute(header, "dimension", config::dim);
            write_attribute(header, "time", time);
            write_attribute(header, "step", step);
            write_attribute(header, "n_global", (int64_t)rows.n_all);
            write_attribute(header, "nranks", mpi::nranks());
            if constexpr (config::astro_physics) {
                write_attribute(header, "UnitLength_in_cm", units.length_in_cm);
                write_attribute(header, "UnitMass_in_g", units.mass_in_g);
                write_attribute(header, "UnitVelocity_in_cm_per_s", units.velocity_in_cm_per_s);
            }
        }
        {
            const Group          mesh_group(H5Gcreate(file, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            std::vector<int64_t> id(n_cells);
            for (int c = 0; c < n_cells; c++)
                id[c] = cells.id[c];
            write_rows(mesh_group, "id", id.data(), rows, 1);
            write_rows(mesh_group, "pos", reinterpret_cast<const double*>(cells.position.data()), rows, config::dim);
            if constexpr (config::moving_mesh) {
                write_rows(mesh_group,
                           "v_mesh",
                           reinterpret_cast<const double*>(cells.mesh_velocity.data()),
                           rows,
                           config::dim);
            }
            if constexpr (config::output_debug) write_mesh_geometry(mesh_group, mesh, ghosts, rows);
        }
        {
            const Group         hydro(H5Gcreate(file, "hydro", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            std::vector<double> energy(n_cells);
            for (int c = 0; c < n_cells; c++)
                energy[c] = gas::energy_density(gas::Primitive{cells.density[c], cells.velocity[c], cells.pressure[c]});
            write_rows(hydro, "rho", cells.density.data(), rows, 1);
            write_rows(hydro, "vel", reinterpret_cast<const double*>(cells.velocity.data()), rows, config::dim);
            write_rows(hydro, "energy", energy.data(), rows, 1);
            write_rows(hydro, "pressure", cells.pressure.data(), rows, 1);
            if constexpr (config::individual_timesteps) write_steps(hydro, cells, steps, step, rows);
        }
        file.close();
        logging::print_labelled("OUTPUT", "Wrote Snapshot %d to %s", number, path.c_str());
    }

    // rank 0 writes the steps and times, which every rank has; every rank its own block of seconds
    void write_profile(const std::string& directory, const std::string& name, const timers::Profile& profile) {
        std::string path = directory;
        if (!path.empty() && path.back() != '/') path += '/';
        path += name;
        const File file(create_by_all_ranks(path));
        if (!file.is_open()) logging::stop_run("OUTPUT: cannot write the profile %s", path.c_str());

        std::string timers, communication;
        for (size_t t = 0; t < profile.paths.size(); t++) {
            timers += profile.paths[t] + "\n";
            communication += std::to_string(profile.communication[t]) + "\n";
        }
        write_attribute(file, "nranks", mpi::nranks());
        write_text_attribute(file, "timers", timers);
        write_text_attribute(file, "communication", communication);

        const long n_rows = (long)profile.steps.size();
        const Rows of_root{0, mpi::is_root() ? n_rows : 0, n_rows};
        write_rows(file, "step", profile.steps.data(), of_root, 1);
        write_rows(file, "time", profile.times.data(), of_root, 1);
        const Rows of_rank{mpi::rank() * n_rows, n_rows, mpi::nranks() * n_rows};
        write_rows(file, "seconds", profile.seconds.data(), of_rank, (int)profile.paths.size());
    }

    namespace {

        // ==========================================================
        // reading
        // ==========================================================

        // HDF5's own messages are turned off; stops the run if the file cannot be opened or has another dimension
        hid_t open_file_of_cells(const char* path, const char* what) {
            H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
            const hid_t file = open_by_all_ranks(path);
            if (file < 0) logging::stop_run("INPUT: cannot open the %s %s", what, path);
            const int dimension = header_number<int>(file, path, "dimension");
            if (dimension != config::dim) {
                logging::stop_run(
                    "INPUT: %s holds cells in %dD, this build is %dD (Config.sh)", path, dimension, config::dim);
            }
            return file;
        }

        template <typename Number> Number header_number(hid_t file, const char* path, const char* name) {
            const Group header(H5Gopen(file, "header", H5P_DEFAULT));
            Number      value{};
            if (!header.is_open() || !read_attribute(header, name, value)) {
                logging::stop_run("INPUT: %s has no header/%s", path, name);
            }
            return value;
        }

        // the rows of the first field; stops the run if another field has another number of rows
        long rows_of_every_field(hid_t file, const char* path, const std::vector<Field>& fields) {
            const long n_all = rows_of(file, path, fields[0].name, fields[0].columns);
            for (const Field& field : fields) {
                if (rows_of(file, path, field.name, field.columns) != n_all) {
                    logging::stop_run("INPUT: the fields of %s do not all have %ld rows", path, n_all);
                }
            }
            return n_all;
        }

        Rows rows_of_this_rank(long n_all) {
            const long r       = mpi::rank();
            const long n_ranks = mpi::nranks();
            const long first   = r * n_all / n_ranks;
            return Rows{first, (r + 1) * n_all / n_ranks - first, n_all};
        }

        // seeds wrapped into the box and rounded to its precision, density and velocity of the rows; collective
        void put_seeds_and_gas_into_cells(hid_t file, const char* path, const Rows& rows, cells::Cells& cells) {
            const std::vector<double> position = read_rows(file, path, "mesh/pos", rows, config::dim);
            const std::vector<double> density  = read_rows(file, path, "hydro/rho", rows, 1);
            const std::vector<double> velocity = read_rows(file, path, "hydro/vel", rows, config::dim);

            cells.n_cells = (int)rows.n;
            cells::make_room_for(cells, cells.n_cells, 0);
            for (long c = 0; c < rows.n; c++) {
                Vec x, v;
                for (int d = 0; d < config::dim; d++) {
                    x[d] = position[config::dim * c + d];
                    v[d] = velocity[config::dim * c + d];
                }
                cells.position[c] = wrapped_into_box(x);
                cells.density[c]  = density[c];
                cells.velocity[c] = v;
            }
        }

        // ==========================================================
        // writing
        // ==========================================================

        std::string snapshot_path(const std::string& directory, int number) {
            std::string path = directory;
            if (!path.empty() && path.back() != '/') path += '/';
            return path + "snapshot_" + std::to_string(number) + ".hdf5";
        }

        // n rows of this rank, after those of the ranks below it; collective
        Rows rows_after_lower_ranks(long n) {
            const std::vector<long long> first_row = first_row_of_every_rank(n);
            return Rows{first_row[mpi::rank()], n, first_row[mpi::nranks()]};
        }

        // volumes, centroids and faces in cell order, each normal pointing to the seed of the neighbour;
        // face_offset and face_neighbor count over the whole file, a ghost cell as the row of its cell on its owner
        void write_mesh_geometry(hid_t                        mesh_group,
                                 const mesh::MeshPointers&    mesh,
                                 const ghosts::GhostPointers& ghosts,
                                 const Rows&                  cell_rows) {
            const int                    n_cells   = mesh.n_cells;
            const std::vector<long long> first_row = first_row_of_every_rank(n_cells);

            std::vector<double> centroid(n_cells * config::dim);
            for (int c = 0; c < n_cells; c++) {
                const Vec absolute = mesh.seed[c] + mesh.centroid_from_seed[c];
                for (int d = 0; d < config::dim; d++)
                    centroid[config::dim * c + d] = absolute[d];
            }

            std::vector<int>     n_faces(n_cells);
            std::vector<int64_t> face_offset(n_cells + 1);
            long                 n_faces_here = 0;
            for (int c = 0; c < n_cells; c++) {
                n_faces[c] = mesh.n_faces[c];
                n_faces_here += n_faces[c];
            }
            const Rows face_rows = rows_after_lower_ranks(n_faces_here);
            long       offset    = 0;
            for (int c = 0; c <= n_cells; c++) {
                face_offset[c] = face_rows.first + offset;
                if (c < n_cells) offset += n_faces[c];
            }

            // every face: the row of its neighbour, its area and normal
            std::vector<int64_t> neighbour(n_faces_here);
            std::vector<double>  area(n_faces_here), normal(n_faces_here * config::dim);
            for (int c = 0; c < n_cells; c++) {
                for (int k = 0; k < n_faces[c]; k++) {
                    const int     f    = mesh.first_face[c] + k;
                    const int64_t out  = face_offset[c] - face_rows.first + k;
                    const int     next = mesh.neighbour[f];
                    const Vec     to   = periodic_difference(mesh.seed[next], mesh.seed[c]);
                    const Vec     unit = to / sqrt(norm2(to));
                    if (next < n_cells) {
                        neighbour[out] = cell_rows.first + next;
                    } else {
                        const int g    = ghosts.ghost_seed[next - n_cells];
                        neighbour[out] = first_row[ghosts.owner[g]] + ghosts.cell_on_owner[g];
                    }
                    area[out] = mesh.area[f];
                    for (int d = 0; d < config::dim; d++)
                        normal[config::dim * out + d] = unit[d];
                }
            }

            const bool last_rank = (mpi::rank() == mpi::nranks() - 1);
            const Rows offset_rows{cell_rows.first, cell_rows.n + (last_rank ? 1 : 0), cell_rows.n_all + 1};
            write_rows(mesh_group, "volume", mesh.volume, cell_rows, 1);
            write_rows(mesh_group, "centroid", centroid.data(), cell_rows, config::dim);
            write_rows(mesh_group, "n_faces", n_faces.data(), cell_rows, 1);
            write_rows(mesh_group, "face_offset", face_offset.data(), offset_rows, 1);
            write_rows(mesh_group, "face_neighbor", neighbour.data(), face_rows, 1);
            write_rows(mesh_group, "face_area", area.data(), face_rows, 1);
            write_rows(mesh_group, "face_normal", normal.data(), face_rows, config::dim);
        }

        // bin and length of the last step, 0 before the first; OUTPUT_DEBUG adds needed bin, planned length, CFL step
        void write_steps(hid_t                          hydro,
                         const cells::Cells&            cells,
                         const timesteps::StepPointers& steps,
                         int                            step,
                         const Rows&                    rows) {
            std::vector<int>    bin(cells.n_cells, 0);
            std::vector<double> length(cells.n_cells, 0.0);
            for (int c = 0; step > 0 && c < cells.n_cells; c++) {
                bin[c]    = cells.steps.bin[c];
                length[c] = (double)(steps.end[c] - steps.start[c]) * steps.tick_length;
            }
            write_rows(hydro, "timebin", bin.data(), rows, 1);
            write_rows(hydro, "timestep", length.data(), rows, 1);

            if constexpr (config::output_debug) {
                std::vector<int>    needed_bin(cells.n_cells, 0);
                std::vector<double> planned(cells.n_cells, 0.0), cfl_step(cells.n_cells, 0.0);
                for (int c = 0; step > 0 && c < cells.n_cells; c++) {
                    needed_bin[c] = cells.steps.needed_bin[c];
                    planned[c]    = (double)(steps.planned_end[c] - steps.start[c]) * steps.tick_length;
                    cfl_step[c]   = cells.cfl_step[c];
                }
                write_rows(hydro, "needed_bin", needed_bin.data(), rows, 1);
                write_rows(hydro, "planned_timestep", planned.data(), rows, 1);
                write_rows(hydro, "cfl_step", cfl_step.data(), rows, 1);
            }
        }

        // first row of every rank, and as the last entry the number of all rows; collective
        std::vector<long long> first_row_of_every_rank(long n) {
            const int              n_ranks = mpi::nranks();
            std::vector<long long> n_of_rank(n_ranks), first_row(n_ranks + 1, 0);
            mpi::gather_from_all_ranks(n, n_of_rank.data());
            for (int r = 0; r < n_ranks; r++)
                first_row[r + 1] = first_row[r] + n_of_rank[r];
            return first_row;
        }

    } // namespace

} // namespace io
