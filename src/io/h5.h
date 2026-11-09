#pragma once

// the HDF5 C API as io uses it: handles that close themselves, and files all ranks open together (included by io.cu)

#include "logging/logging.h"
#include "mpi/mpi.h"

#include <hdf5.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace io {

    // an HDF5 object that closes itself at the end of its scope
    template <herr_t (*CLOSE)(hid_t)> class Handle {
      public:
        explicit Handle(hid_t id) : id(id) {}
        ~Handle() {
            if (id >= 0) CLOSE(id);
        }
        Handle(const Handle&)            = delete;
        Handle& operator=(const Handle&) = delete;

        operator hid_t() const { return id; }
        bool is_open() const { return id >= 0; }

        void close() {
            if (id >= 0) CLOSE(id);
            id = -1;
        }

      private:
        hid_t id;
    };
    using File         = Handle<H5Fclose>;
    using Group        = Handle<H5Gclose>;
    using Dataset      = Handle<H5Dclose>;
    using Space        = Handle<H5Sclose>;
    using Attribute    = Handle<H5Aclose>;
    using PropertyList = Handle<H5Pclose>;
    using Datatype     = Handle<H5Tclose>;

    // the block of rows of this rank in a dataset of n_all rows: n rows from first on
    struct Rows {
        long first;
        long n;
        long n_all;
    };

    inline hid_t access_by_all_ranks() {
        const hid_t properties = H5Pcreate(H5P_FILE_ACCESS);
#ifdef USE_MPI
        H5Pset_fapl_mpio(properties, mpi::communicator(), MPI_INFO_NULL);
#endif
        return properties;
    }

    inline hid_t transfer_by_all_ranks() {
        const hid_t properties = H5Pcreate(H5P_DATASET_XFER);
#ifdef USE_MPI
        H5Pset_dxpl_mpio(properties, H5FD_MPIO_COLLECTIVE);
#endif
        return properties;
    }

    // for reading; a missing file gives a handle that is not open; collective
    inline hid_t open_by_all_ranks(const char* path) {
        const PropertyList access(access_by_all_ranks());
        return H5Fopen(path, H5F_ACC_RDONLY, access);
    }

    // a new file, or one written over; collective
    inline hid_t create_by_all_ranks(const std::string& path) {
        const PropertyList access(access_by_all_ranks());
        return H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, access);
    }

    inline hid_t hdf5_type_of(const double*) {
        return H5T_NATIVE_DOUBLE;
    }
    inline hid_t hdf5_type_of(const int*) {
        return H5T_NATIVE_INT;
    }
    inline hid_t hdf5_type_of(const int64_t*) {
        return H5T_NATIVE_INT64;
    }

    // number of rows of a dataset of columns numbers per row; stops the run if it is missing or has another shape
    inline long rows_of(hid_t file, const char* path, const char* name, int columns) {
        const Dataset dataset(H5Dopen(file, name, H5P_DEFAULT));
        if (!dataset.is_open()) logging::stop_run("INPUT: %s has no %s", path, name);

        const Space space(H5Dget_space(dataset));
        hsize_t     shape[2]   = {0, 1};
        const int   dimensions = H5Sget_simple_extent_ndims(space);
        if (dimensions == 1 || dimensions == 2) H5Sget_simple_extent_dims(space, shape, nullptr);
        if (!(dimensions == 1 || dimensions == 2) || shape[1] != (hsize_t)columns) {
            logging::stop_run("INPUT: %s in %s has not %d number(s) per row", name, path, columns);
        }
        return (long)shape[0];
    }

    // one number of an object; false if it has none of that name
    template <typename Number> bool read_attribute(hid_t object, const char* name, Number& value) {
        const Attribute attribute(H5Aopen(object, name, H5P_DEFAULT));
        return attribute.is_open() && H5Aread(attribute, hdf5_type_of(&value), &value) >= 0;
    }

    // the block of rows of this rank; collective
    template <typename Number = double>
    std::vector<Number> read_rows(hid_t file, const char* path, const char* name, const Rows& rows, int columns) {
        const Dataset dataset(H5Dopen(file, name, H5P_DEFAULT));
        const Space   in_file(H5Dget_space(dataset));
        const hsize_t first[2]   = {(hsize_t)rows.first, 0};
        const hsize_t count[2]   = {(hsize_t)rows.n, (hsize_t)columns};
        const int     dimensions = H5Sget_simple_extent_ndims(in_file);
        const Space   in_memory(H5Screate_simple(dimensions, count, nullptr));
        if (rows.n > 0) {
            H5Sselect_hyperslab(in_file, H5S_SELECT_SET, first, nullptr, count, nullptr);
        } else {
            H5Sselect_none(in_file);
            H5Sselect_none(in_memory);
        }

        // a rank without rows still takes part, with a buffer of one row
        std::vector<Number> values(rows.n * columns + columns);
        const PropertyList  transfer(transfer_by_all_ranks());
        if (H5Dread(dataset, hdf5_type_of(values.data()), in_memory, in_file, transfer, values.data()) < 0) {
            logging::stop_run("INPUT: cannot read %s from %s", name, path);
        }
        values.resize(rows.n * columns);
        return values;
    }

    // every rank writes the same number; collective
    template <typename Number> void write_attribute(hid_t object, const char* name, Number value) {
        const Space     space(H5Screate(H5S_SCALAR));
        const Attribute attribute(H5Acreate2(object, name, hdf5_type_of(&value), space, H5P_DEFAULT, H5P_DEFAULT));
        if (!attribute.is_open() || H5Awrite(attribute, hdf5_type_of(&value), &value) < 0) {
            logging::stop_run("OUTPUT: cannot write the attribute %s", name);
        }
    }

    // every rank writes the same text; collective
    inline void write_text_attribute(hid_t object, const char* name, const std::string& text) {
        const Datatype type(H5Tcopy(H5T_C_S1));
        H5Tset_size(type, std::max<size_t>(text.size(), 1));
        const Space     space(H5Screate(H5S_SCALAR));
        const Attribute attribute(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT));
        if (!attribute.is_open() || H5Awrite(attribute, type, text.c_str()) < 0) {
            logging::stop_run("OUTPUT: cannot write the attribute %s", name);
        }
    }

    // a dataset of rows.n_all rows, one dimension for one column; this rank writes its block from values; collective
    template <typename Number>
    void write_rows(hid_t group, const char* name, const Number* values, const Rows& rows, int columns) {
        const int     dimensions = (columns == 1) ? 1 : 2;
        const hsize_t shape[2]   = {(hsize_t)rows.n_all, (hsize_t)columns};
        const hsize_t first[2]   = {(hsize_t)rows.first, 0};
        const hsize_t count[2]   = {(hsize_t)rows.n, (hsize_t)columns};
        const Space   in_file(H5Screate_simple(dimensions, shape, nullptr));
        const Dataset dataset(
            H5Dcreate2(group, name, hdf5_type_of(values), in_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if (!dataset.is_open()) logging::stop_run("OUTPUT: cannot make the dataset %s", name);

        const Space in_memory(H5Screate_simple(dimensions, count, nullptr));
        if (rows.n > 0) {
            H5Sselect_hyperslab(in_file, H5S_SELECT_SET, first, nullptr, count, nullptr);
        } else {
            H5Sselect_none(in_file);
            H5Sselect_none(in_memory);
        }

        // a rank without rows still takes part, with a buffer that is not null
        const Number       nothing{};
        const PropertyList transfer(transfer_by_all_ranks());
        if (H5Dwrite(dataset, hdf5_type_of(values), in_memory, in_file, transfer, (rows.n > 0) ? values : &nothing) <
            0) {
            logging::stop_run("OUTPUT: cannot write the dataset %s", name);
        }
    }

} // namespace io
