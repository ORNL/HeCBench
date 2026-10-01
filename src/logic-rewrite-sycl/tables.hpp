#include <sycl/sycl.hpp>
#pragma once
#include "common.h"

/**
 * Functions for vector of vectors, where the vector elements can have different lengths.
 * The implementation is by using a 2D array, and oversized vectors occupy
 * more than one row.
 **/
namespace VecVecLink {

template <typename T, int TABLE_NUM_ROWS, int TABLE_NUM_COLS>
inline void writeElem(T *vTable, int *vLinks, int *pnNextRow,
                               int *pCurrRow, int *pCurrCol, const T elem,
                               int *pCounter = NULL) {
    if (*pCurrCol == TABLE_NUM_COLS) {
        // expand a new row
        int lastRow = *pCurrRow;
        *pCurrRow =
            nsycl::atomic_fetch_add(
                pnNextRow, 1);
        assert(*pCurrRow < TABLE_NUM_ROWS);
        vLinks[lastRow] = *pCurrRow;
        vLinks[*pCurrRow] = 0;
        *pCurrCol = 0;
    }
    vTable[(*pCurrRow) * TABLE_NUM_COLS + (*pCurrCol)] = elem;
    (*pCurrCol)++;
    
    if (pCounter != NULL)
        (*pCounter)++;
}

template <typename T, int TABLE_NUM_ROWS, int TABLE_NUM_COLS>
inline T readElem(T *vTable, int *vLinks, int *pCurrRow,
                           int *pCurrCol) {
    if (*pCurrCol == TABLE_NUM_COLS) {
        // expand a new row
        *pCurrCol = 0;
        *pCurrRow = vLinks[*pCurrRow];
    }
    return vTable[(*pCurrRow) * TABLE_NUM_COLS + (*pCurrCol)++];
}

} // namespace VecVecLink


/**
 * Functions for 2D arrays (Tables), which can be regarded as vector of vectors 
 * where the vector elements are of the same length.
 **/
namespace Table {

/**
 * Gather the vectors given by vIndices in vTable to a consecutive array vArray, 
 * where the index ranges of each vector is given in vRanges.
 **/
template <typename T, int TABLE_NUM_COLS>
void gatherTableToConsecutive(const T * vTable, const int * vTableLens, 
                                         const int * vIndices, const int * vRanges, 
                                         T * vArray, int nIndices) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nIndices) {
        int nodeId = vIndices[idx];
        const T * vTableEntry = &vTable[nodeId * TABLE_NUM_COLS];
        int startIdx = (idx == 0 ? 0 : vRanges[idx - 1]);
        int length = vTableLens[nodeId];

        assert(length == vRanges[idx] - startIdx);

        for (int i = 0; i < length; i++) {
            vArray[startIdx + i] = vTableEntry[i];
        }
    }
}

} // namespace Table
