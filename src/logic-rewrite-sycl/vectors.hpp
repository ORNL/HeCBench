#include <sycl/sycl.hpp>
#pragma once

template <typename T, int nCap>
struct VecsMem {
    static_assert(nCap <= (1 << 16), "The capacity of VecsMem should be no larger than 2^16!\n");

    inline T *fetch(int nWords) {
        if (nWords <= 0)
            return NULL;
        if (nSize + nWords > nCap) {
            // (device code) capacity exceeded -> try to decrease K in refactor
            assert(0);
            return NULL;
        }
        nSize += nWords;
        return pArray + nSize - nWords;
    }

    inline void shrink(int nSizeNew) {
        assert(nSize >= nSizeNew);
        nSize = nSizeNew;
    }

    int nSize = 0;
    T pArray[nCap];
};
