#pragma once
#include <sycl/sycl.hpp>
#include <tuple>
#include "common.h"
#include "hash_table.h"

namespace Aig {

SYCL_EXTERNAL void buildHashTable(const int *pFanin0, const int *pFanin1,
                                  uint64 *htKeys, uint32 *htValues,
                                  int htCapacity, int nNodes, int nPIs);

std::tuple<int, int *, int *, int *, int *, int>
strash(const int * pFanin0, const int * pFanin1, const int * pOuts, int * pNumFanouts,
       int nObjs, int nPIs, int nPOs);

/**
 * Check and retrieve AND node id from the hashtable, considering trivial cases.
 * Note that this function assumes that fanin0/1 already exist in the hashtable,
 * otherwise the returned trivial cases might not be correct. 
 **/
inline uint32 retrieveHashTableCheckTrivial(int faninLit0,
                                                     int faninLit1,
                                                     const uint64 *htKeys,
                                                     const uint32 *htValues,
                                                     int htCapacity) {
    uint64 key;
    uint32 res = checkTrivialAndCases(faninLit0, faninLit1);
    if (res != HASHTABLE_EMPTY_VALUE<uint64, uint32>)
        return res;
    
    key = formAndNodeKey(faninLit0, faninLit1);
    res = retrieve_single<uint64, uint32>(htKeys, htValues, key, htCapacity);
    return res;
}

} // namespace Aig
