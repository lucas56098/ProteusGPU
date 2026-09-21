// writes profile.hdf5 (included by profiler.cu)

namespace {

    // profile.hdf5 handles, closed by hand (see close_profile_log)
    hid_t s_file       = -1;
    bool  s_log_active = false;
    int   s_my_rank    = 0;
    int   s_nranks     = 1;

    hid_t   s_per_step = -1;
    hid_t   s_cum      = -1;
    hid_t   s_names    = -1;
    hid_t   s_kinds    = -1;
    hsize_t s_rows     = 0;

    std::unordered_map<std::string, size_t> s_timer_index; // row of a timer in the file

    std::vector<long long> s_prev_cum; // last cumulative per row, for the difference

    // what this rank reported last
    std::vector<std::string> s_sent_names;
    std::vector<char>        s_sent_kinds;
    std::vector<size_t>      s_sent_slots;

    // kind as the 3 byte string in the file
    static const char* kind_str(char k) {
        switch (k) {
        case 'm':
            return "mpi";
        case 'g':
            return "gpu";
        default:
            return "cpu";
        }
    }

    // chunk [16 steps, nranks, 256 timers], name 256 bytes
    constexpr hsize_t PROFILE_STEP_CHUNK  = 16;
    constexpr hsize_t PROFILE_TIMER_CHUNK = 256;
    constexpr size_t  PROFILE_NAME_LEN    = 256;
    constexpr size_t  PROFILE_KIND_LEN    = 3;

    // more than one rank: MPI-IO
    bool parallel_log() {
#ifdef USE_MPI
        return s_nranks > 1;
#else
        return false;
#endif
    }

    // string type of the two name lists
    h5::Type fixed_string(size_t len) {
        h5::Type t(H5Tcopy(H5T_C_S1));
        H5Tset_size(t, len);
        H5Tset_strpad(t, H5T_STR_NULLPAD);
        return t;
    }

    // [step, rank, timer] table, grows in both
    hid_t create_table(const char* name) {
        hsize_t   dims[3]  = {0, (hsize_t)s_nranks, 0};
        hsize_t   max[3]   = {H5S_UNLIMITED, (hsize_t)s_nranks, H5S_UNLIMITED};
        hsize_t   chunk[3] = {PROFILE_STEP_CHUNK, (hsize_t)s_nranks, PROFILE_TIMER_CHUNK};
        h5::Space space(H5Screate_simple(3, dims, max));
        h5::Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
        H5Pset_chunk(dcpl, 3, chunk);
        // the default fill would make all ranks zero the chunk first
        if (parallel_log()) H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
        return H5Dcreate(s_file, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }

    // growing list of fixed length strings
    hid_t create_list(const char* name, size_t len) {
        hsize_t   dims = 0, max = H5S_UNLIMITED, chunk = PROFILE_TIMER_CHUNK;
        h5::Space space(H5Screate_simple(1, &dims, &max));
        h5::Plist dcpl(H5Pcreate(H5P_DATASET_CREATE));
        H5Pset_chunk(dcpl, 1, &chunk);
        h5::Type type = fixed_string(len);
        return H5Dcreate(s_file, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }

    // reads names or kinds back on a restart
    std::vector<std::string> read_list(hid_t dset, size_t len) {
        std::vector<std::string> out;
        h5::Space                space(H5Dget_space(dset));
        const hssize_t           n = H5Sget_simple_extent_npoints(space);
        if (n <= 0) return out;
        std::vector<char> buf((size_t)n * len);
        h5::Type          type = fixed_string(len);
        if (H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()) < 0) return out;
        for (hssize_t i = 0; i < n; i++) {
            const char* s = buf.data() + (size_t)i * len;
            out.emplace_back(s, strnlen(s, len));
        }
        return out;
    }

    // both tables keep the same shape
    void resize_tables(hsize_t rows, hsize_t ntimers) {
        hsize_t dims[3] = {rows, (hsize_t)s_nranks, ntimers};
        H5Dset_extent(s_per_step, dims);
        H5Dset_extent(s_cum, dims);
    }

    // closed by hand, like the file
    void close_datasets() {
        for (hid_t* d : {&s_per_step, &s_cum, &s_names, &s_kinds}) {
            if (*d >= 0) H5Dclose(*d);
            *d = -1;
        }
    }

    // give a timer its row and starting value
    void register_timer(const std::string& name) {
        auto it             = s_restart_baseline.find(name);
        s_timer_index[name] = s_prev_cum.size();
        s_prev_cum.push_back(it != s_restart_baseline.end() ? it->second : 0);
    }

    // appended, so a row never moves, restart included
    void append_timers(const std::vector<std::string>& names, const std::vector<char>& kinds) {
        const hsize_t old_n = s_prev_cum.size();
        const hsize_t add   = names.size();
        const hsize_t total = old_n + add;

        std::vector<char> nbuf(add * PROFILE_NAME_LEN, '\0');
        std::vector<char> kbuf(add * PROFILE_KIND_LEN, '\0');
        for (size_t i = 0; i < add; i++) {
            std::memcpy(&nbuf[i * PROFILE_NAME_LEN], names[i].data(), names[i].size());
            std::memcpy(&kbuf[i * PROFILE_KIND_LEN], kind_str(kinds[i]), PROFILE_KIND_LEN);
        }

        h5::Plist dxpl(H5Pcreate(H5P_DATASET_XFER));
#ifdef USE_MPI
        if (parallel_log()) H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
#endif
        for (int k = 0; k < 2; k++) {
            const hid_t  dset = k ? s_kinds : s_names;
            const size_t len  = k ? PROFILE_KIND_LEN : PROFILE_NAME_LEN;
            H5Dset_extent(dset, &total);
            h5::Space fspace(H5Dget_space(dset));
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &old_n, NULL, &add, NULL);
            h5::Space mspace(H5Screate_simple(1, &add, NULL));
            h5::Type  type = fixed_string(len);
            H5Dwrite(dset, type, mspace, fspace, dxpl, k ? kbuf.data() : nbuf.data());
        }
        resize_tables(s_rows, total);
        for (const auto& n : names)
            register_timer(n);
    }

    // append every name the file does not have yet
    void add_new_timers(const std::vector<std::string>& my_names, const std::vector<char>& my_kinds) {
        const auto all_names = allgather_timer_names(my_names, s_nranks);
        const auto all_kinds = allgather_timer_kinds(my_kinds, s_nranks);

        std::map<std::string, char> fresh;
        for (int r = 0; r < s_nranks; r++) {
            for (size_t i = 0; i < all_names[r].size(); i++) {
                const std::string& n = all_names[r][i];
                if (s_timer_index.count(n)) continue;
                const char k  = (i < all_kinds[r].size()) ? all_kinds[r][i] : 'c';
                auto       it = fresh.find(n);
                if (it == fresh.end() || it->second == 'c') fresh[n] = k;
            }
        }
        if (fresh.empty()) return;

        std::vector<std::string> names;
        std::vector<char>        kinds;
        for (const auto& kv : fresh) {
            if (kv.first.size() > PROFILE_NAME_LEN) {
                proteus_mpi::exit_failure(
                    "PROFILER: timer name longer than %zu bytes: %s\n", PROFILE_NAME_LEN, kv.first.c_str());
            }
            names.push_back(kv.first);
            kinds.push_back(kv.second);
        }
        append_timers(names, kinds);
    }

    // this rank's [step, rank] block of one table
    void write_block(hid_t dset, int step, const std::vector<double>& values) {
        if (values.empty()) return;
        h5::Space fspace(H5Dget_space(dset));
        hsize_t   start[3] = {(hsize_t)step, (hsize_t)s_my_rank, 0};
        hsize_t   count[3] = {1, 1, values.size()};
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
        h5::Space mspace(H5Screate_simple(3, count, NULL));
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, values.data());
    }

    // continue an existing log if layout and rank count match
    bool open_existing_log(int restart_step) {
        if (H5Lexists(s_file, "timer_names", H5P_DEFAULT) <= 0 || H5Lexists(s_file, "timer_kinds", H5P_DEFAULT) <= 0 ||
            H5Lexists(s_file, "per_step", H5P_DEFAULT) <= 0 || H5Lexists(s_file, "cumulative", H5P_DEFAULT) <= 0) {
            return false;
        }
        s_names    = H5Dopen(s_file, "timer_names", H5P_DEFAULT);
        s_kinds    = H5Dopen(s_file, "timer_kinds", H5P_DEFAULT);
        s_per_step = H5Dopen(s_file, "per_step", H5P_DEFAULT);
        s_cum      = H5Dopen(s_file, "cumulative", H5P_DEFAULT);
        bool ok    = s_names >= 0 && s_kinds >= 0 && s_per_step >= 0 && s_cum >= 0;

        hsize_t dims[3] = {0, 0, 0}, dims_cum[3] = {0, 0, 0};
        if (ok) {
            h5::Space sp(H5Dget_space(s_per_step));
            h5::Space sc(H5Dget_space(s_cum));
            ok = H5Sget_simple_extent_ndims(sp) == 3 && H5Sget_simple_extent_ndims(sc) == 3;
            if (ok) {
                H5Sget_simple_extent_dims(sp, dims, NULL);
                H5Sget_simple_extent_dims(sc, dims_cum, NULL);
            }
        }
        std::vector<std::string> names;
        if (ok) {
            names            = read_list(s_names, PROFILE_NAME_LEN);
            const auto kinds = read_list(s_kinds, PROFILE_KIND_LEN);
            ok = dims[1] == (hsize_t)s_nranks && dims[2] == names.size() && kinds.size() == names.size() &&
                 dims_cum[1] == dims[1] && dims_cum[2] == dims[2];
        }
        if (!ok) {
            close_datasets();
            return false;
        }

        for (const auto& n : names)
            register_timer(n);
        // cut the tables back to the resumed step
        s_rows = (hsize_t)restart_step;
        resize_tables(s_rows, names.size());
        return true;
    }

} // namespace

// continue on a restart, else a new file; MPI-IO from two ranks up
void Profiler::open_profile_log(const std::string& path, int restart_step) {
    s_my_rank    = proteus_mpi::rank();
    s_nranks     = proteus_mpi::nranks();
    s_log_active = true;

    h5::Plist fapl(H5Pcreate(H5P_FILE_ACCESS));
#ifdef USE_MPI
    if (parallel_log() && H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL) < 0) {
        proteus_mpi::exit_failure("PROFILER: could not select the MPI-IO driver for %s\n", path.c_str());
    }
#endif

    if (restart_step >= 0) {
        s_file = H5Fopen(path.c_str(), H5F_ACC_RDWR, fapl);
        if (s_file >= 0 && !open_existing_log(restart_step)) {
            logging::root() << "PROFILER: " << path << " has another layout or rank count. Starting a new log."
                            << std::endl;
            H5Fclose(s_file);
            s_file = -1;
        }
    }
    if (s_file < 0) {
        s_file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        if (s_file < 0) {
            logging::root() << "PROFILER: could not create " << path << ". Running without a profile log." << std::endl;
            s_log_active = false;
            return;
        }
        s_per_step = create_table("per_step");
        s_cum      = create_table("cumulative");
        s_names    = create_list("timer_names", PROFILE_NAME_LEN);
        s_kinds    = create_list("timer_kinds", PROFILE_KIND_LEN);
        s_rows     = 0;
    }

    H5Fflush(s_file, H5F_SCOPE_GLOBAL);
}

// closes the log at the end of the run
// by hand: a destructor would run after HDF5's atexit handler
void Profiler::close_profile_log() {
    if (!s_log_active) return;
    s_log_active = false;
    if (s_file < 0) return;
    close_datasets();
    H5Fclose(s_file);
    s_file = -1;
}

// error path: no HDF5 under MPI, H5Fclose is collective and would block
void Profiler::abort_profile_log() {

    if (parallel_log()) {
        s_log_active = false;
        return;
    }
    close_profile_log();
}

// one row per step: totals and the difference to the row before
void Profiler::log_timestep(int step) {
    if (!s_log_active) return;

    auto rows = collect_current();
    // sorted, so all ranks propose new timers in the same order
    std::sort(rows.begin(),
              rows.end(),
              [](const std::pair<std::string, long long>& a, const std::pair<std::string, long long>& b) {
                  return a.first < b.first;
              });
    std::vector<std::string> my_names;
    std::vector<char>        my_kinds;
    std::vector<long long>   my_vals;
    my_names.reserve(rows.size());
    my_kinds.reserve(rows.size());
    my_vals.reserve(rows.size());
    for (const auto& r : rows) {
        my_names.push_back(r.first);
        auto it = s_kind.find(r.first);
        my_kinds.push_back(it != s_kind.end() ? it->second : 'c');
        my_vals.push_back(r.second);
    }

    // growing the tables is collective, so all ranks must agree
    int changed = (my_names != s_sent_names || my_kinds != s_sent_kinds) ? 1 : 0;
#ifdef USE_MPI
    if (parallel_log()) MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (changed) {
        add_new_timers(my_names, my_kinds);
        s_sent_names = my_names;
        s_sent_kinds = my_kinds;
        s_sent_slots.resize(my_names.size());
        for (size_t j = 0; j < my_names.size(); j++)
            s_sent_slots[j] = s_timer_index[my_names[j]];
    }

    const size_t ntimers = s_prev_cum.size();
    if ((hsize_t)step + 1 > s_rows) {
        s_rows = (hsize_t)step + 1;
        resize_tables(s_rows, ntimers);
    }

    std::vector<long long> cum_us(ntimers, 0);
    for (size_t j = 0; j < my_vals.size(); j++)
        cum_us[s_sent_slots[j]] = my_vals[j];

    std::vector<double> per_step(ntimers), cumulative(ntimers);
    for (size_t i = 0; i < ntimers; i++) {
        per_step[i]   = (cum_us[i] - s_prev_cum[i]) / 1e6;
        cumulative[i] = cum_us[i] / 1e6;
        s_prev_cum[i] = cum_us[i];
    }
    write_block(s_per_step, step, per_step);
    write_block(s_cum, step, cumulative);

    H5Fflush(s_file, H5F_SCOPE_GLOBAL);
}
