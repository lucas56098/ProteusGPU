#ifndef IO_H5_H
#define IO_H5_H

// Wraps the HDF5 C API in handles that close themselves, plus typed read / write helpers.

#include "hdf5.h"
#include <cstdint>
#include <iostream>
#include <vector>

namespace h5 {

    // hsize_t is an HDF5 typedef and not always uint64_t, only the width is fixed;
    // so pass it by value here and never take a hsize_t* or hsize_t& in this interface
    static_assert(sizeof(hsize_t) == sizeof(uint64_t), "hsize_t is expected to be 64 bits wide");

    // ==========================================================
    // handles
    // ==========================================================

    // owns one hid_t and closes it at the end of the scope; movable, not copyable
    template <herr_t (*CLOSE)(hid_t)> class Handle {
      public:
        Handle() : m_id(H5I_INVALID_HID) {}
        explicit Handle(hid_t id) : m_id(id) {}
        // exit() does not run destructors, so leave every h5 scope before aborting a run
        ~Handle() { reset(); }

        Handle(const Handle&)            = delete;
        Handle& operator=(const Handle&) = delete;

        Handle(Handle&& other) : m_id(other.m_id) { other.m_id = H5I_INVALID_HID; }
        Handle& operator=(Handle&& other) {
            if (this != &other) {
                reset();
                m_id       = other.m_id;
                other.m_id = H5I_INVALID_HID;
            }
            return *this;
        }

        operator hid_t() const { return m_id; }

        bool valid() const { return m_id >= 0; }

        void reset() {
            if (m_id >= 0) { CLOSE(m_id); }
            m_id = H5I_INVALID_HID;
        }

      private:
        hid_t m_id;
    };

    // one alias per HDF5 object kind
    using File    = Handle<H5Fclose>;
    using Group   = Handle<H5Gclose>;
    using Space   = Handle<H5Sclose>;
    using Dataset = Handle<H5Dclose>;
    using Attr    = Handle<H5Aclose>;
    using Plist   = Handle<H5Pclose>;
    using Type    = Handle<H5Tclose>;

    // C++ type -> H5T_NATIVE_* id
    template <typename T> struct native;
    template <> struct native<double> {
        static hid_t id() { return H5T_NATIVE_DOUBLE; }
    };
    template <> struct native<int> {
        static hid_t id() { return H5T_NATIVE_INT; }
    };
    template <> struct native<int64_t> {
        static hid_t id() { return H5T_NATIVE_INT64; }
    };

    // checks that a dataset has the expected rank
    inline bool check_rank(hid_t space, const char* name, int expected) {
        const int ndims = H5Sget_simple_extent_ndims(space);
        if (ndims != expected) {
            std::cerr << "H5: Error! Dataset '" << name << "' has rank " << ndims << ", expected " << expected
                      << std::endl;
            return false;
        }
        return true;
    }

    // ==========================================================
    // write
    // ==========================================================

    // writes one scalar attribute
    template <typename T> inline bool write_attr(hid_t parent, const char* name, T value) {
        const hid_t type = native<T>::id();
        Space       space(H5Screate(H5S_SCALAR));
        Attr        attr(H5Acreate(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT));
        if (!attr.valid() || H5Awrite(attr, type, &value) < 0) {
            std::cerr << "H5: Error! Could not write attribute '" << name << "'" << std::endl;
            return false;
        }
        return true;
    }

    // creates a dataset of the given shape and writes it
    template <typename T>
    inline bool write_dataset(hid_t parent, const char* name, const T* data, int rank, const hsize_t* dims) {
        const hid_t type = native<T>::id();
        Space       space(H5Screate_simple(rank, dims, NULL));
        Dataset     dset(H5Dcreate(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if (!dset.valid()) {
            std::cerr << "H5: Error! Could not create dataset '" << name << "'" << std::endl;
            return false;
        }
        if (H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
            std::cerr << "H5: Error! Could not write dataset '" << name << "'" << std::endl;
            return false;
        }
        return true;
    }

    // writes n values
    template <typename T> inline bool write_dataset_1d(hid_t parent, const char* name, const T* data, hsize_t n) {
        const hsize_t dims[1] = {n};
        return write_dataset(parent, name, data, 1, dims);
    }

    // writes n rows of dim values
    template <typename T>
    inline bool write_dataset_2d(hid_t parent, const char* name, const T* data, hsize_t n, hsize_t dim) {
        const hsize_t dims[2] = {n, dim};
        return write_dataset(parent, name, data, 2, dims);
    }

    // ==========================================================
    // read
    // ==========================================================

    // reads one scalar attribute
    template <typename T> inline bool read_attr(hid_t parent, const char* name, T& out) {
        const hid_t type = native<T>::id();
        Attr        attr(H5Aopen(parent, name, H5P_DEFAULT));
        if (!attr.valid() || H5Aread(attr, type, &out) < 0) {
            std::cerr << "H5: Error! Could not read attribute '" << name << "'" << std::endl;
            return false;
        }
        return true;
    }

    // reads a whole 1D dataset into out (resized to the dataset size)
    template <typename T> inline bool read_dataset_1d(hid_t parent, const char* name, std::vector<T>& out) {
        const hid_t type = native<T>::id();
        Dataset     dset(H5Dopen(parent, name, H5P_DEFAULT));
        if (!dset.valid()) {
            std::cerr << "H5: Error! Could not open dataset '" << name << "'" << std::endl;
            return false;
        }
        Space space(H5Dget_space(dset));
        if (!check_rank(space, name, 1)) { return false; }
        hsize_t dim = 0;
        H5Sget_simple_extent_dims(space, &dim, NULL);
        out.resize(dim);
        if (H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0) {
            std::cerr << "H5: Error! Could not read dataset '" << name << "'" << std::endl;
            return false;
        }
        return true;
    }

    // reads a whole 2D dataset into out as flat rows; out_rows gets the row count
    template <typename T>
    inline bool read_dataset_2d(hid_t parent, const char* name, std::vector<T>& out, uint64_t* out_rows = NULL) {
        const hid_t type = native<T>::id();
        Dataset     dset(H5Dopen(parent, name, H5P_DEFAULT));
        if (!dset.valid()) {
            std::cerr << "H5: Error! Could not open dataset '" << name << "'" << std::endl;
            return false;
        }
        Space space(H5Dget_space(dset));
        if (!check_rank(space, name, 2)) { return false; }
        hsize_t dims[2] = {0, 0};
        H5Sget_simple_extent_dims(space, dims, NULL);
        out.resize(dims[0] * dims[1]);
        if (H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0) {
            std::cerr << "H5: Error! Could not read dataset '" << name << "'" << std::endl;
            return false;
        }
        if (out_rows) { *out_rows = (uint64_t)dims[0]; }
        return true;
    }

#ifdef USE_MPI

    // ==========================================================
    // parallel read (USE_MPI)
    // ==========================================================

    // reads rows [row_lo, row_lo + n_local) of a 1D dataset
    template <typename T>
    inline bool
    read_hyperslab_1d(hid_t parent, const char* name, hsize_t row_lo, hsize_t n_local, std::vector<T>& out) {
        const hid_t type = native<T>::id();
        Dataset     dset(H5Dopen(parent, name, H5P_DEFAULT));
        if (!dset.valid()) {
            std::cerr << "H5: Error! Could not open dataset '" << name << "'" << std::endl;
            return false;
        }
        Space filespace(H5Dget_space(dset));
        if (!check_rank(filespace, name, 1)) { return false; }
        hsize_t offset = row_lo;
        hsize_t count  = n_local;

        // a rank without rows still takes part, it just selects nothing
        if (n_local > 0) {
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL);
        } else {
            H5Sselect_none(filespace);
        }
        Space memspace(H5Screate_simple(1, &count, NULL));
        if (n_local == 0) H5Sselect_none(memspace);

        out.resize(n_local);

        // all ranks read in one call
        Plist dxpl(H5Pcreate(H5P_DATASET_XFER));
        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
        return H5Dread(dset, type, memspace, filespace, dxpl, out.data()) >= 0;
    }

    // same for a 2D dataset (trailing dim must be expected_dim)
    template <typename T>
    inline bool read_hyperslab_2d(
        hid_t parent, const char* name, hsize_t row_lo, hsize_t n_local, hsize_t expected_dim, std::vector<T>& out) {
        const hid_t type = native<T>::id();
        Dataset     dset(H5Dopen(parent, name, H5P_DEFAULT));
        if (!dset.valid()) {
            std::cerr << "H5: Error! Could not open dataset '" << name << "'" << std::endl;
            return false;
        }
        Space filespace(H5Dget_space(dset));
        if (!check_rank(filespace, name, 2)) { return false; }
        hsize_t full_dims[2] = {0, 0};
        H5Sget_simple_extent_dims(filespace, full_dims, NULL);
        if (full_dims[1] != expected_dim) {
            std::cerr << "H5: Error! Dataset '" << name << "' has trailing dim " << full_dims[1] << ", expected "
                      << expected_dim << std::endl;
            return false;
        }

        hsize_t offset[2] = {row_lo, 0};
        hsize_t count[2]  = {n_local, expected_dim};
        if (n_local > 0) {
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
        } else {
            H5Sselect_none(filespace);
        }
        Space memspace(H5Screate_simple(2, count, NULL));
        if (n_local == 0) H5Sselect_none(memspace);

        out.resize(n_local * expected_dim);

        // all ranks read in one call
        Plist dxpl(H5Pcreate(H5P_DATASET_XFER));
        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
        return H5Dread(dset, type, memspace, filespace, dxpl, out.data()) >= 0;
    }

#endif

} // namespace h5

#endif
