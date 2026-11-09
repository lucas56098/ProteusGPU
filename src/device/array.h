#pragma once

// the Array class and the packing of items (included by device.h)

#include <cstring>

namespace device {

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
        ElementType* to_fill = elements;
        parallel_for(length, [=] HD(long i) { to_fill[i] = value; });
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
        const ElementType* from = field.data();
        char*              to   = items.data();
        parallel_for(n, [=] HD(long i) { memcpy(to + i * item_bytes + offset, &from[index[i]], sizeof(ElementType)); });
    }

    template <typename ElementType>
    void unpack_from_items(const Array<char>&  items,
                           long                n,
                           int                 item_bytes,
                           int                 offset,
                           long                first,
                           Array<ElementType>& field) {
        const char*  from = items.data();
        ElementType* to   = field.data() + first;
        parallel_for(n, [=] HD(long i) { memcpy(&to[i], from + i * item_bytes + offset, sizeof(ElementType)); });
    }

    template <typename ElementType>
    void unpack_from_items_at(const Array<char>&  items,
                              long                n,
                              int                 item_bytes,
                              int                 offset,
                              const int*          index,
                              Array<ElementType>& field) {
        const char*  from = items.data();
        ElementType* to   = field.data();
        parallel_for(n, [=] HD(long i) { memcpy(&to[index[i]], from + i * item_bytes + offset, sizeof(ElementType)); });
    }

} // namespace device
