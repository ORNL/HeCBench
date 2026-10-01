#pragma once
#include <sycl/sycl.hpp>
#include "common.h"
#include "hash_table.h"

namespace Aig {

template <int CUT_TABLE_NUM_COLS, int STACK_SIZE, bool useHashtable = false>
inline int
findReconvMFFCCut_iter(const int rootId, const int *pFanin0, const int *pFanin1,
                       const int *pNumFanouts, const int *pLevels,
                       int *vCutTable, int *vCutSizes, int *travIds,
                       int *travRefs, int *pTravSize, int *pConeSize, int nPIs,
                       int nMaxCutSize) {
    int nodeId, faninId, bestId = -1, bestIdx = -1, faninLoc;
    int bestCost = 100, currCost;
    int fFanin0Visited, fFanin1Visited;
    int nodeRef;
    int * vCutRoot = &vCutTable[rootId * CUT_TABLE_NUM_COLS];


    // 1. evaluate each node in the cut and choose the best one
    for (int i = 0; i < vCutSizes[rootId]; i++) {
        nodeId = vCutRoot[i];

        // get ref value of nodeId
        nodeRef = HASHTABLE_EMPTY_VALUE<int, int>;
        if constexpr (useHashtable) {
            nodeRef = st_map_query<int, int>(travIds, travRefs, nodeId, STACK_SIZE);
        } else {
            for (int j = 0; j < *pTravSize; j++)
                if (travIds[j] == nodeId) {
                    nodeRef = travRefs[j];
                    break;
                }
        }
        assert(nodeRef != (HASHTABLE_EMPTY_VALUE<int, int>));
        
        // get the number of new leaves
        if (dUtils::AigIsPIConst(nodeId, nPIs) || nodeRef != 0) {
            // stop at PI/const or non-MFFC nodes
            currCost = 999;
        } else {
            fFanin0Visited = fFanin1Visited = 0;

            faninId = dUtils::AigNodeID(pFanin0[nodeId]);
            if constexpr (useHashtable) {
                if (st_set_locate<int>(travIds, faninId, STACK_SIZE) != -1)
                    fFanin0Visited = 1;
            } else {
                for (int j = 0; j < *pTravSize; j++)
                    if (travIds[j] == faninId) {
                        fFanin0Visited = 1;
                        break;
                    }
            }

            faninId = dUtils::AigNodeID(pFanin1[nodeId]);
            if constexpr (useHashtable) {
                if (st_set_locate<int>(travIds, faninId, STACK_SIZE) != -1)
                    fFanin1Visited = 1;
            } else {
                for (int j = 0; j < *pTravSize; j++)
                    if (travIds[j] == faninId) {
                        fFanin1Visited = 1;
                        break;
                    }
            }

            currCost = (1 - fFanin0Visited) + (1 - fFanin1Visited);
        }

        // update best node
        if (bestCost > currCost || (bestCost == currCost && pLevels[nodeId] > pLevels[bestId]))
            bestCost = currCost, bestId = nodeId, bestIdx = i;
        if (bestCost == 0)
            break;
    }

    if (bestId == -1)
        return 0;
    assert(bestCost < 3);

    if (vCutSizes[rootId] - 1 + bestCost > nMaxCutSize)
        return 0;
    assert(dUtils::AigIsNode(bestId, nPIs));

    // 2. expand bestId
    // remove the bestId from cut list
    for (int i = bestIdx + 1; i < vCutSizes[rootId]; i++)
        vCutRoot[i - 1] = vCutRoot[i];
    vCutSizes[rootId]--;

    // check whether the fanins of bestId is visited or not
    // if not, add into the cut list
    faninId = dUtils::AigNodeID(pFanin0[bestId]);
    if constexpr (useHashtable) {
        faninLoc = st_set_locate<int>(travIds, faninId, STACK_SIZE);
    } else {
        faninLoc = -1;
        for (int j = 0; j < *pTravSize; j++)
            if (travIds[j] == faninId) {
                faninLoc = j;
                break;
            }
    }
    if (faninLoc == -1) {
        // not visited
        vCutRoot[vCutSizes[rootId]++] = faninId;
        // add into visited and record ref
        if constexpr (useHashtable) {
            st_map_insert_or_query<int, int>(travIds, travRefs, faninId, pNumFanouts[faninId], STACK_SIZE);
            faninLoc = st_set_locate<int>(travIds, faninId, STACK_SIZE);
        } else {
            assert(*pTravSize < STACK_SIZE);
            faninLoc = *pTravSize;
            travIds[*pTravSize] = faninId;
            travRefs[(*pTravSize)++] = pNumFanouts[faninId];
        }
        assert(faninLoc != -1);
    }
    // decrement ref
    travRefs[faninLoc]--;

    faninId = dUtils::AigNodeID(pFanin1[bestId]);
    if constexpr (useHashtable) {
        faninLoc = st_set_locate<int>(travIds, faninId, STACK_SIZE);
    } else {
        faninLoc = -1;
        for (int j = 0; j < *pTravSize; j++)
            if (travIds[j] == faninId) {
                faninLoc = j;
                break;
            }
    }
    if (faninLoc == -1) {
        // not visited
        vCutRoot[vCutSizes[rootId]++] = faninId;
        // add into visited and record ref
        if constexpr (useHashtable) {
            st_map_insert_or_query<int, int>(travIds, travRefs, faninId, pNumFanouts[faninId], STACK_SIZE);
            faninLoc = st_set_locate<int>(travIds, faninId, STACK_SIZE);
        } else {
            assert(*pTravSize < STACK_SIZE);
            faninLoc = *pTravSize;
            travIds[*pTravSize] = faninId;
            travRefs[(*pTravSize)++] = pNumFanouts[faninId];
        }
        assert(faninLoc != -1);
    }
    // decrement ref
    travRefs[faninLoc]--;

    // increment cone size
    (*pConeSize)++;

    assert(vCutSizes[rootId] <= nMaxCutSize);
    return 1;
}


/**
 * Find a reconvergence driven cut that consist of only MFFC nodes.
 * Returns the size (nSaved) of the cone. 
 **/
template <int CUT_TABLE_NUM_COLS, int STACK_SIZE, bool useHashtable = false>
inline int
findReconvMFFCCut(const int rootId, const int *pFanin0, const int *pFanin1,
                  const int *pNumFanouts, const int *pLevels, int *vCutTable,
                  int *vCutSizes, int nPIs, int nMaxCutSize) {
    int faninId;
    int travIds[STACK_SIZE], travRefs[STACK_SIZE];
    int travSize = 0;
    int coneSize = 0;
    int * vCutRoot = &vCutTable[rootId * CUT_TABLE_NUM_COLS];
    
    // note, visited contains exactly the same elements as travIds,
    // so there is no need to create a separate one

    // the MFFC of rootId should not be checked twice
    assert(vCutSizes[rootId] == -1);
    vCutSizes[rootId] = 0;

    // initialize the cut list and ref/visited list by adding the root and its two fanins
    assert(dUtils::AigIsNode(rootId, nPIs));
    if constexpr (useHashtable) {
        st_map_clear<int, int>(travIds, travRefs, STACK_SIZE);
        st_map_insert_or_query<int, int>(travIds, travRefs, rootId, 0, STACK_SIZE);
    } else {
        travIds[travSize] = rootId;
        travRefs[travSize++] = 0;
    }

    faninId = dUtils::AigNodeID(pFanin0[rootId]);
    vCutRoot[vCutSizes[rootId]++] = faninId;
    if constexpr (useHashtable) {
        st_map_insert_or_query<int, int>(travIds, travRefs, faninId, pNumFanouts[faninId] - 1, STACK_SIZE);
    } else {
        travIds[travSize] = faninId;
        travRefs[travSize++] = pNumFanouts[faninId] - 1;
    }

    faninId = dUtils::AigNodeID(pFanin1[rootId]);
    vCutRoot[vCutSizes[rootId]++] = faninId;
    if constexpr (useHashtable) {
        st_map_insert_or_query<int, int>(travIds, travRefs, faninId, pNumFanouts[faninId] - 1, STACK_SIZE);
    } else {
        travIds[travSize] = faninId;
        travRefs[travSize++] = pNumFanouts[faninId] - 1;
    }
    coneSize = 1;

    // iteratively expand the cut
    while (findReconvMFFCCut_iter<CUT_TABLE_NUM_COLS, STACK_SIZE, useHashtable>(
        rootId, pFanin0, pFanin1, pNumFanouts, pLevels, 
        vCutTable, vCutSizes, travIds, travRefs, &travSize, 
        &coneSize, nPIs, nMaxCutSize
    ));
    assert(vCutSizes[rootId] <= nMaxCutSize);

    return coneSize;
}





} // namespace Aig
