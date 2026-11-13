#pragma once

// the Array class and the packing of items (included by device.h)

#include <cstring>

namespace device {

    // The loops of these templates are named functors, not __host__ __device__ lambdas: nvcc wraps such a lambda in a
    // type that several .cu files share at link time, while each file keeps its own pointer to the host function
    // behind it, so a lambda of a template used in two files can call the null pointer of the other one.
    namespace array_loops {

        template <typename ElementType> struct Fill {
            ElementType* to;
            ElementType  value;
            HD void operator()(long i) const { to[i] = value; }
        };

        template <typename ElementType> struct Pack {
            const ElementType* from;
            const int*         index;
            char*              to;
            int                item_bytes;
            int                offset;
            HD void operator()(long i) const {
                memcpy(to + i * item_bytes + offset, &from[index[i]], sizeof(ElementType));
            }
        };

        template <typename ElementType> struct Unpack {
            const char*  from;
            ElementType* to;
            int          item_bytes;
            int          offset;
            HD void operator()(long i) const { memcpy(&to[i], from + i * item_bytes + offset, sizeof(ElementType)); }
        };

        template <typename ElementType> struct UnpackAt {
            const char*  from;
            const int*   index;
            ElementType* to;
            int          item_bytes;
            int          offset;
            HD void operator()(long i) const {
                memcpy(&to[index[i]], from + i * item_bytes + offset, sizeof(ElementType));
            }
        };

    } // namespace array_loops

    template <typename ElementType>
    Array<ElementType>::Array(const char* name, long size)
        : name(name), length(size),
          elements(static_cast<ElementType*>(allocate_bytes(name, size * (long)sizeof(ElementType)))) {}

    template <typename ElementType> Array<ElementType>::~Array() {
        free_bytes(elements, length * (long)sizeof(ElementType));
    }

    // a quarter more than asked when it grows again, so it rarely grows a third time
    template <typename ElementType> void Array<ElementType>::grow_to_at_least(long size) {
        if (size <= length) return;
        Array larger(name, (length == 0) ? size : size + size / 4);
        copy_bytes(larger.elements, elements, length * (long)sizeof(ElementType));
        swap(larger);
    }

    template <typename ElementType> void Array<ElementType>::fill(const ElementType& value) {
        parallel_for(length, array_loops::Fill<ElementType>{elements, value});
    }

    template <typename ElementType> void copy(const Array<ElementType>& from, Array<ElementType>& to, long n) {
        copy_bytes(to.data(), from.data(), n * (long)sizeof(ElementType));
    }

    template <typename ElementType>
    void pack_into_items(const Array<ElementType>& field,
                         const int*                index,
                         long                      n,
                         int                       item_bytes,
                         int                       offset,
                         Array<char>&              items) {
        parallel_for(n, array_loops::Pack<ElementType>{field.data(), index, items.data(), item_bytes, offset});
    }

    template <typename ElementType>
    void unpack_from_items(const Array<char>&  items,
                           long                n,
                           int                 item_bytes,
                           int                 offset,
                           long                first,
                           Array<ElementType>& field) {
        parallel_for(n, array_loops::Unpack<ElementType>{items.data(), field.data() + first, item_bytes, offset});
    }

    template <typename ElementType>
    void unpack_from_items_at(const Array<char>&  items,
                              long                n,
                              int                 item_bytes,
                              int                 offset,
                              const int*          index,
                              Array<ElementType>& field) {
        parallel_for(n, array_loops::UnpackAt<ElementType>{items.data(), index, field.data(), item_bytes, offset});
    }

} // namespace device
