#ifndef IO_H5_H
#define IO_H5_H

#include "hdf5.h"
#include <cstdint>
#include <iostream>

// Thin layer over the HDF5 API.

// NOTE: exit() does not run destructors of local objects, so a fatal error must leave every
//       h5:: scope before it exits or the file is never flushed and the snapshot is
//       truncated. Report the failure up to the caller and exit there.
namespace h5 {

    template <herr_t (*CLOSE)(hid_t)> class Handle {
      public:
        Handle() : m_id(H5I_INVALID_HID) {}
        explicit Handle(hid_t id) : m_id(id) {}
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

    using File    = Handle<H5Fclose>;
    using Group   = Handle<H5Gclose>;
    using Space   = Handle<H5Sclose>;
    using Dataset = Handle<H5Dclose>;
    using Attr    = Handle<H5Aclose>;

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

    // ------------------------------------------------------------
    // writing
    // ------------------------------------------------------------

    template <typename T> inline void write_attr(hid_t parent, const char* name, T value) {
        const hid_t type = native<T>::id();
        Space       space(H5Screate(H5S_SCALAR));
        Attr        attr(H5Acreate(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT));
        H5Awrite(attr, type, &value);
    }

    template <typename T>
    inline bool write_dataset(hid_t parent, const char* name, const T* data, int rank, const hsize_t* dims) {
        const hid_t type = native<T>::id();
        Space       space(H5Screate_simple(rank, dims, NULL));
        Dataset     dset(H5Dcreate(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if (!dset.valid()) {
            std::cerr << "H5: Error! Could not create dataset '" << name << "'" << std::endl;
            return false;
        }
        H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        return true;
    }

    template <typename T> inline bool write_dataset_1d(hid_t parent, const char* name, const T* data, hsize_t n) {
        const hsize_t dims[1] = {n};
        return write_dataset(parent, name, data, 1, dims);
    }

    template <typename T>
    inline bool write_dataset_2d(hid_t parent, const char* name, const T* data, hsize_t n, hsize_t dim) {
        const hsize_t dims[2] = {n, dim};
        return write_dataset(parent, name, data, 2, dims);
    }

} // namespace h5

#endif // IO_H5_H
