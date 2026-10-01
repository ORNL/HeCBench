#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <tuple>
#include <vector>
#include <ctime>
#include <climits>

#include "refactor.h"
#include "mffc.hpp"
#include "truth.hpp"
#include "strash.hpp"
#include "tables.hpp"
#include "common.h"
#include "alg_factor.hpp"
#include <time.h>

using namespace sop;

struct isSmallMFFC {
    
    bool operator()(const int elem) const {
        return elem != -1 && elem < 2;
    }
};

struct bitwiseNot {
    
    unsigned operator()(const unsigned elem) const {
        return ~elem;
    }
};

struct identity {
    template <typename T>
    T operator()(const T& x) const { return x; }
};


// debug functions
void printMffcCut(int * vCutTable, int * vCutSizes, int * vConeSizes,
                             const int * pFanin0, const int * pFanin1, 
                             int nNodes, int nPIs, int nPOs,
                             const sycl::stream &stream_ct1) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx != 0)
        return;

    int counter = 0;
    for (int i = 0; i < nNodes; i++) {
        int id = i + nPIs + 1;
        if (vCutSizes[id] == -1)
            continue;
        counter++;

        stream_ct1 << "root: %d, cone size: %d | ";
        for (int j = 0; j < vCutSizes[id]; j++) {
            stream_ct1 << "%d ";
        }
        stream_ct1 << "\n";
    }
    stream_ct1 << "Total number of MFFCs: %d\n";
}


template <bool useHashtable = false>
void recordMFFC(const int * vRoots, 
                           const int * pFanin0, const int * pFanin1, 
                           const int * pNumFanouts, const int * pLevels, 
                           int * vCutTable, int * vCutSizes, int * vConeSizes, 
                           int nPIs, int nMaxCutSize, int nRoots) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int nThreads = item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    int idx = item_ct1.get_global_id(2);
    int rootId, nSaved;

    for (; idx < nRoots; idx += nThreads) {
        rootId = vRoots[idx];

        nSaved = Aig::findReconvMFFCCut<CUT_TABLE_SIZE, STACK_SIZE, useHashtable>(
            rootId, pFanin0, pFanin1, pNumFanouts, pLevels, 
            vCutTable, vCutSizes, nPIs, nMaxCutSize
        );
        vConeSizes[rootId] = nSaved;
    }
}

void setStatus(const int * vRoots,
                          const int * vCutTable, const int * vCutSizes,
                          int * vNodesStatus,
                          int nPIs, int nRoots) {
    // vNodesStatus should be set to all zeros before launching this kernel
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int nThreads = item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    int idx = item_ct1.get_global_id(2);
    int rootId, nodeId, nCutSize;

    for (; idx < nRoots; idx += nThreads) {
        rootId = vRoots[idx];

        const int * vCutRoot = &vCutTable[rootId * CUT_TABLE_SIZE];
        nCutSize = vCutSizes[rootId];
        for (int i = 0; i < nCutSize; i++) {
            nodeId = vCutRoot[i];
            // update the status array if the MFFC rooted at the node is not explored
            if (dUtils::AigIsNode(nodeId, nPIs) && vCutSizes[nodeId] == -1)
                vNodesStatus[nodeId] = 1;
        }
    }
}

void getCutTruthRanges(const int * vResynRoots, const int * vCutSizes, 
                                  int * vCutRanges, int * vTruthRanges, int nResyn) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nResyn) {
        int nodeId = vResynRoots[idx];
        int cutSize = vCutSizes[nodeId];
        assert(cutSize > 0);
        vCutRanges[idx] = cutSize;
        vTruthRanges[idx] = dUtils::TruthWordNum(cutSize);
    }
}

int evalSubgNumAdded(int rootId, int *pNewRootLevel, const int *vCurrCut,
                     int nVars, int nNodeMax, int nLevelMax, const int *pLevels,
                     const int *vNode2ConeResynIdx, const uint64 *htKeys,
                     const uint32 *htValues, int htCapacity,
                     const uint64 *vSubgTable, const int *vSubgLinks,
                     const int *vSubgLens, int subgIdx) {
    int i, counter, temp;
    int lit0, lit1, id0, id1, func0, func1, currId, fCompRoot, newLevel;
    uint32 retrId;
    int vFuncs[SUBG_CAP], vLevels[SUBG_CAP];
    int length = vSubgLens[subgIdx] + nVars;
    int currRowIdx, columnPtr;

    assert(vSubgLens[subgIdx] > 0);

    // check the case of the resyned cut is a const or a single var of cut nodes
    if (vSubgLens[subgIdx] == 1) {
        subgUtil::unbindAndNodeKeyFlag(vSubgTable[subgIdx * SUBG_TABLE_SIZE], &lit0, &lit1, &fCompRoot);
        if (lit0 == lit1)
            return 0;
    }

    // initialize funcs (ids) and levels for the leaves
    for (i = 0; i < nVars; i++) {
        vFuncs[i] = vCurrCut[i];
        vLevels[i] = pLevels[vCurrCut[i]];
    }

    counter = 0;
    currRowIdx = subgIdx, columnPtr = 0;
    for (i = nVars; i < length; i++) {
        if (columnPtr == SUBG_TABLE_SIZE) {
            // expand a new row
            columnPtr = 0;
            currRowIdx = vSubgLinks[currRowIdx];
        }
        // get the children of the current subgraph node
        subgUtil::unbindAndNodeKeyFlag(
            vSubgTable[currRowIdx * SUBG_TABLE_SIZE + (columnPtr++)], 
            &lit0, &lit1, &fCompRoot
        );
        assert(lit0 < lit1);
        id0 = dUtils::AigNodeID(lit0), id1 = dUtils::AigNodeID(lit1);
        assert(id0 < i && id1 < i);
        func0 = vFuncs[id0], func1 = vFuncs[id1]; // ids of its children in the original AIG

        // if they are both present, find the resulting node in hashtable
        if (func0 != -1 && func1 != -1) {
            func0 = dUtils::AigNodeLitCond(func0, dUtils::AigNodeIsComplement(lit0));
            func1 = dUtils::AigNodeLitCond(func1, dUtils::AigNodeIsComplement(lit1));
            if (func0 > func1) // though they are properly ordered in subgraph id, in AIG id they may not
                temp = func0, func0 = func1, func1 = temp;
            // if (func0 >= func1) {
            //     printf("func0: %d, func1: %d\n", func0, func1);
            //     printf("  cuts: ");
            //     for (int j = 0; j < nVars; j++)
            //         printf("%d ", vCurrCut[j]);
            //     printf("\n  subg: ");
            //     int currRowIdx0 = subgIdx, columnPtr0 = 0;
            //     for (int j = 0; j < vSubgLens[subgIdx]; j++) {
            //         if (columnPtr0 == SUBG_TABLE_SIZE) {
            //             // expand a new row
            //             columnPtr0 = 0;
            //             currRowIdx0 = vSubgLinks[currRowIdx0];
            //         }
            //         subgUtil::unbindAndNodeKeyFlag(
            //             vSubgTable[currRowIdx0 * SUBG_TABLE_SIZE + (columnPtr0++)], 
            //             &lit0, &lit1, &fCompRoot
            //         );
            //         printf("%s%d,%s%d ", dUtils::AigNodeIsComplement(lit0) ? "!" : "", dUtils::AigNodeID(lit0),
            //                              dUtils::AigNodeIsComplement(lit1) ? "!" : "", dUtils::AigNodeID(lit1));
            //     }
            //     printf("\n");
            // }
            assert(func0 <= func1);

            retrId = Aig::retrieveHashTableCheckTrivial(func0, func1, htKeys, htValues, htCapacity);
            if (retrId == (HASHTABLE_EMPTY_VALUE<uint64, uint32>))
                currId = -1;
            else
                currId = (int)retrId;
            // return -1 if the node is the same as the original root
            if (retrId == (uint32)rootId)
                return -1;
        } else
            currId = -1;
        
        // count one new node
        // nodes whose vNode2ConeResynIdx are assigned are MFFC nodes and are to be removed,
        // so do not count shareable logic with them
        if (currId == -1 || vNode2ConeResynIdx[currId] != -1) {
            if (++counter > nNodeMax)
                return -1;
        }
        // count new level
        newLevel = 1 + sycl::max(vLevels[id0], vLevels[id1]);
        if (currId != -1) {
            // previously went though the hashtable retrival
            if (currId == 0) // const 0/1
                newLevel = 0;
            else if (currId == dUtils::AigNodeID(func0))
                newLevel = vLevels[id0];
            else if (currId == dUtils::AigNodeID(func1))
                newLevel = vLevels[id1];
        }
        if (newLevel > nLevelMax)
            return -1;
        
        // save new func (id) and level
        vFuncs[i] = currId;
        vLevels[i] = newLevel;
    }
    *pNewRootLevel = vLevels[length - 1];
    return counter;
}

void evalFactoredForm(const int * vResynRoots, const int * vCuts, const int * vCutRanges,
                                 const int * vNumSaved, const int * pLevels, const int * vNode2ConeResynIdx, 
                                 const uint64 * htKeys, const uint32 * htValues, int htCapacity,
                                 const uint64 * vSubgTable, const int * vSubgLinks, const int * vSubgLens, 
                                 int * vSelectedSubgInd, int nResyn) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int nThreads = item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    int idx = item_ct1.get_global_id(2);
    int coneIdx, rootId;
    int nVars, nSaved, nAdded, nOtherAdded, nNewLevel, nOtherNewLevel;
    int startIdx, endIdx;
    unsigned warpMask;
    int fSelectedSubg;

    for (; idx < 2 * nResyn; idx += nThreads) {
        warpMask = 0xffffffff;
        coneIdx = idx >> 1;
        rootId = vResynRoots[coneIdx];
        
        startIdx = (coneIdx == 0 ? 0 : vCutRanges[coneIdx - 1]);
        endIdx = vCutRanges[coneIdx];
        nVars = endIdx - startIdx;
        nSaved = vNumSaved[rootId];

        nAdded = evalSubgNumAdded(
            rootId, &nNewLevel, vCuts + startIdx, 
            nVars, nSaved, 1000000000, pLevels, vNode2ConeResynIdx, 
            htKeys, htValues, htCapacity, vSubgTable, vSubgLinks, vSubgLens, idx
        );
        // exchange with the paired (negated) subgraph lane via sub-group xor shuffle
        nOtherAdded = sycl::permute_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), nAdded,
            1); // nAdded of the corresponding negated subgraph
        nOtherNewLevel = sycl::permute_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), nNewLevel, 1);

        if (idx % 2 == 0) {
            // select a better subgraph among the pair
            fSelectedSubg = -1;
            if (nAdded > -1)
                fSelectedSubg = 0;
            if (nOtherAdded > -1) {
                if (nAdded == -1)
                    fSelectedSubg = 1;
                else if (nOtherAdded < nAdded || (nOtherAdded == nAdded && 
                         (vSubgLens[idx + 1] == 1 || nOtherNewLevel < nNewLevel)))
                    fSelectedSubg = 1;
            }

            // write to vSelectedSubgInd
            vSelectedSubgInd[coneIdx] = (fSelectedSubg == -1 ? -1 : idx + fSelectedSubg);
        }
    }

}

void duplicateHashTableWithoutMFFCs(const int * vNode2ConeResynIdx, const int * vSelectedSubgInd,
                                               const uint64 * htKeys, const uint32 * htValues, int htCapacity,
                                               uint64 * htDestKeys, uint32 * htDestValues, int htDestCapacity) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < htCapacity && htKeys[idx] != HASHTABLE_EMPTY_KEY<uint64, uint32>) {
        int nodeId = htValues[idx];
        int coneResynIdx = vNode2ConeResynIdx[nodeId];
        // in two cases the cone nodes should be kept: 
        // (1) the cone is not a explored MFFC, (2) the corresponding two resyned graphs are not better than
        // the original cone (i.e. vSelectedSubgInd == -1)
        if (coneResynIdx == -1 || vSelectedSubgInd[coneResynIdx] == -1) {
            insert_single_no_update<uint64, uint32>(htDestKeys, htDestValues, 
                                                    htKeys[idx], htValues[idx], htDestCapacity);
        }
    }
}

void checkSingleVarSubg(const uint64 * vSubgTable, const int * vSubgLinks, const int * vSubgLens,
                                   const int * vResynRoots, const int * vSelectedSubgInd, int * vOldRoot2NewRootLits, 
                                   int * vFinishedMark, int nObjs, int nResyn) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nResyn) {
        int subgIdx = vSelectedSubgInd[idx];
        int rootId = vResynRoots[idx];
        int lit0, lit1, fComp;
        
        if (subgIdx == -1) {
            // both of the two graphs are not better than the original one
            vFinishedMark[idx] = 1;
        } else if (vSubgLens[subgIdx] == 1) {
            // take care of the case that the resyned cut is a const or a single var of cut nodes
            subgUtil::unbindAndNodeKeyFlag(vSubgTable[subgIdx * SUBG_TABLE_SIZE], &lit0, &lit1, &fComp);
            if (lit0 == lit1) {
                assert(dUtils::AigNodeID(lit0) < nObjs); // in this case lit0 is using the AIG id
                assert(fComp == dUtils::AigNodeIsComplement(lit0));

                vOldRoot2NewRootLits[rootId] = lit0;
                vFinishedMark[idx] = 1;
            }
        }
    }
}

void insertSubgIter(int iter, const int *vResynIdSeq, const int *vCuts,
                    const int *vCutRanges, uint64 *htDestKeys,
                    uint32 *htDestValues, int htDestCapacity,
                    uint64 *vSubgTable, const int *vSubgLinks,
                    const int *vSubgLens, const int *vSelectedSubgInd,
                    int idCounter, int nReplace) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nReplace) {
        int startIdx, endIdx, nVars;
        int lit0, lit1, id0, id1, fanin0, fanin1, fComp, temp;
        uint32 temp0, temp1;
        uint64 key;
        int currRowIdx, columnPtr;
        int subgRows[SUBG_CAP / SUBG_TABLE_SIZE];
        int resynIdx = vResynIdSeq[idx];
        int subgIdx = vSelectedSubgInd[resynIdx];

        startIdx = (resynIdx == 0 ? 0 : vCutRanges[resynIdx - 1]);
        endIdx = vCutRanges[resynIdx];
        nVars = endIdx - startIdx;
        const int * vCurrCut = vCuts + startIdx;

        assert(iter < vSubgLens[subgIdx]);
        assert(iter < SUBG_CAP);
        // fetch the iter-th node of the subgraph
        currRowIdx = subgIdx, columnPtr = iter % SUBG_TABLE_SIZE;
        subgRows[0] = currRowIdx;
        for (int i = 0; i < (iter / SUBG_TABLE_SIZE); i++) {
            currRowIdx = vSubgLinks[currRowIdx];
            subgRows[i + 1] = currRowIdx;
        }
        subgUtil::unbindAndNodeKeyFlag(vSubgTable[currRowIdx * SUBG_TABLE_SIZE + columnPtr], 
                                       &lit0, &lit1, &fComp);
        id0 = dUtils::AigNodeID(lit0), id1 = dUtils::AigNodeID(lit1);

        // convert lit0/1 into AIG ids
        if (id0 < nVars) {
            fanin0 = vCurrCut[id0]; // cut saves id
            fanin0 = dUtils::AigNodeLitCond(fanin0, dUtils::AigNodeIsComplement(lit0));
        } else {
            id0 -= nVars;
            assert(id0 < iter);
            currRowIdx = subgRows[id0 / SUBG_TABLE_SIZE], columnPtr = id0 % SUBG_TABLE_SIZE;
            unbindAndNodeKeys(vSubgTable[currRowIdx * SUBG_TABLE_SIZE + columnPtr], &temp0, &temp1);
            assert(temp0 == 0); // has already been processed in previous iterations
            fanin0 = (int)temp1; // temp1 saves lit instead of id
            fanin0 = dUtils::AigNodeNotCond(fanin0, dUtils::AigNodeIsComplement(lit0));
        }
        if (id1 < nVars) {
            fanin1 = vCurrCut[id1]; // cut saves id
            fanin1 = dUtils::AigNodeLitCond(fanin1, dUtils::AigNodeIsComplement(lit1));
        } else {
            id1 -= nVars;
            assert(id1 < iter);
            currRowIdx = subgRows[id1 / SUBG_TABLE_SIZE], columnPtr = id1 % SUBG_TABLE_SIZE;
            unbindAndNodeKeys(vSubgTable[currRowIdx * SUBG_TABLE_SIZE + columnPtr], &temp0, &temp1);
            assert(temp0 == 0); // has already been processed in previous iterations
            fanin1 = (int)temp1; // temp1 saves lit instead of id
            fanin1 = dUtils::AigNodeNotCond(fanin1, dUtils::AigNodeIsComplement(lit1));
        }
        if (fanin0 > fanin1) // though they are properly ordered in subgraph id, in AIG id they may not
            temp = fanin0, fanin0 = fanin1, fanin1 = temp;

        // check trivial
        temp0 = checkTrivialAndCases(fanin0, fanin1);
        if (temp0 == HASHTABLE_EMPTY_VALUE<uint64, uint32>) {
            // non-trivial, insert into hashtable
            assert(fanin0 < fanin1);
            key = formAndNodeKey(fanin0, fanin1);
            // assign new (tentative) id as idCounter + idx, which is unique
            insert_single_no_update<uint64, uint32>(htDestKeys, htDestValues, key, 
                                                    (uint32)(idCounter + idx), htDestCapacity);
        }
        // save the converted key into the corresponding location in vSubgTable
        key = subgUtil::formAndNodeKeyFlag(fanin0, fanin1, fComp);
        currRowIdx = subgRows[iter / SUBG_TABLE_SIZE], columnPtr = iter % SUBG_TABLE_SIZE;
        vSubgTable[currRowIdx * SUBG_TABLE_SIZE + columnPtr] = key;
    }
}

void updateInsertedIdsIter(int iter, const int * vResynRoots, const int * vResynIdSeq,
                                      const uint64 * htDestKeys, const uint32 * htDestValues, int htDestCapacity,
                                      uint64 * vSubgTable, const int * vSubgLinks, const int * vSubgLens,
                                      const int * vSelectedSubgInd, int * vOldRoot2NewRootLits, int * vFinishedMark, 
                                      int nReplace) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nReplace) {
        int fanin0, fanin1, fComp;
        uint32 temp0;
        uint64 key;
        int currRowIdx, columnPtr;
        int resynIdx = vResynIdSeq[idx];
        int subgIdx = vSelectedSubgInd[resynIdx];
        int oldRootId = vResynRoots[resynIdx];

        assert(iter < vSubgLens[subgIdx]);
        assert(iter < SUBG_CAP);
        // fetch the iter-th node of the subgraph
        currRowIdx = subgIdx, columnPtr = iter % SUBG_TABLE_SIZE;
        for (int i = 0; i < (iter / SUBG_TABLE_SIZE); i++)
            currRowIdx = vSubgLinks[currRowIdx];
        subgUtil::unbindAndNodeKeyFlag(vSubgTable[currRowIdx * SUBG_TABLE_SIZE + columnPtr], 
                                       &fanin0, &fanin1, &fComp);
        
        // check trivial
        temp0 = checkTrivialAndCases(fanin0, fanin1);
        if (temp0 == HASHTABLE_EMPTY_VALUE<uint64, uint32>) {
            // non-trivial, retrieve the corresponding id from hashtable
            assert(fanin0 < fanin1);
            key = formAndNodeKey(fanin0, fanin1);
            temp0 = retrieve_single<uint64, uint32>(htDestKeys, htDestValues, key, htDestCapacity); // id
            temp0 = temp0 << 1; // convert to lit
        }
        // save the updated literal of current node into the corresponding location in vSubgTable
        // mark the first entry as 0 to indicate that this key represents the literal of current node
        key = formAndNodeKey(0, temp0);
        vSubgTable[currRowIdx * SUBG_TABLE_SIZE + columnPtr] = key;

        // deal with finished subgraphs
        if (iter == vSubgLens[subgIdx] - 1) {
            vFinishedMark[idx] = 1;
            // save the new root literal
            vOldRoot2NewRootLits[oldRootId] = dUtils::AigNodeNotCond(temp0, fComp);
        }
    }
}

void checkInsertion(const int * vResynRoots, const int * vOldRoot2NewRootLits, 
                               const int * vSelectedSubgInd, int nResyn) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nResyn) {
        int subgIdx = vSelectedSubgInd[idx];
        int rootId = vResynRoots[idx];
        if (subgIdx != -1) {
            assert(vOldRoot2NewRootLits[rootId] != -1);
        }
    }
}

void unbindKeysUpdateOldRoots(const int * vOldRoot2NewRootLits, 
                                         const uint64 * vReconstructedKeys, const uint32 * vReconstructedIds,
                                         int * vFanin0New, int * vFanin1New, int nEntries, int nObjs, int nBufferLen) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nEntries) {
        uint32 lit0, lit1, nodeId;
        int newLit0, newLit1;
        unbindAndNodeKeys(vReconstructedKeys[idx], &lit0, &lit1);
        nodeId = vReconstructedIds[idx];
        assert(nodeId < nBufferLen);
        if (nodeId < nObjs && vOldRoot2NewRootLits[nodeId] != -1) {
            // nodeId is a reconstructed old root
            // convert this old root node into a buffer of the new root literal
            newLit0 = dUtils::AigConst1;
            newLit1 = vOldRoot2NewRootLits[nodeId];
        } else {
            newLit0 = (int)lit0, newLit1 = (int)lit1;
        }

        vFanin0New[nodeId] = newLit0;
        vFanin1New[nodeId] = newLit1;
    }
}

int insertMFFCs(uint64 *htDestKeys, uint32 *htDestValues, int htDestCapacity,
                uint64 *vSubgTable, int *vSubgLinks, int *vSubgLens,
                const int *vResynRoots, const int *vCuts, const int *vCutRanges,
                const int *vSelectedSubgInd, int *vOldRoot2NewRootLits,
                int nObjs, int nResyn) {
    sycl::queue &q_ct1 = nsycl::queue();
    // create a sequence of resyn indices and shrink it iteratively,
    // but do not change vSelectedSubgInd since it is aligned with vResynRoots
    int * vResynIdSeq, * vFinishedMark, * pNewListEnd;
    vResynIdSeq = sycl::malloc_device<int>(nResyn, q_ct1);
    vFinishedMark = sycl::malloc_device<int>(nResyn, q_ct1);
    q_ct1.memset(vFinishedMark, 0, nResyn * sizeof(int)).wait();
    nsycl::iota(q_ct1, vResynIdSeq,
               vResynIdSeq + nResyn);

    // process the single var subgraphs first
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nResyn, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            checkSingleVarSubg(vSubgTable, vSubgLinks, vSubgLens, vResynRoots,
                               vSelectedSubgInd, vOldRoot2NewRootLits,
                               vFinishedMark, nObjs, nResyn);
        });
    q_ct1.wait();
    pNewListEnd = nsycl::remove_if(q_ct1, vResynIdSeq,
        vResynIdSeq + nResyn, vFinishedMark, identity{});
    assert(pNewListEnd - vResynIdSeq <= nResyn);
    int nReplace = pNewListEnd - vResynIdSeq;
    printf("Number of subgraphs to be inserted: %d\n", nReplace);

    int iter = 0; // the index of subgraph nodes that are being processed in the current iteration
    int idCounter = nObjs; // used for assigning new tentative ids of inserted nodes
    while (nReplace > 0) {
        q_ct1.memset(vFinishedMark, 0, nReplace * sizeof(int)).wait();
        q_ct1.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(nReplace, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                insertSubgIter(iter, vResynIdSeq, vCuts, vCutRanges, htDestKeys,
                               htDestValues, htDestCapacity, vSubgTable,
                               vSubgLinks, vSubgLens, vSelectedSubgInd,
                               idCounter, nReplace);
            });
        q_ct1.wait();
        q_ct1.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(nReplace, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                updateInsertedIdsIter(iter, vResynRoots, vResynIdSeq,
                                      htDestKeys, htDestValues, htDestCapacity,
                                      vSubgTable, vSubgLinks, vSubgLens,
                                      vSelectedSubgInd, vOldRoot2NewRootLits,
                                      vFinishedMark, nReplace);
            });
        q_ct1.wait();
        // increment idCounter
        assert(idCounter + nReplace < (INT_MAX / 2));
        idCounter += nReplace;

        // shrink according to vFinishedMark
        pNewListEnd = nsycl::remove_if(q_ct1, vResynIdSeq,
            vResynIdSeq + nReplace, vFinishedMark, identity{});
        q_ct1.wait();
        assert(pNewListEnd - vResynIdSeq <= nReplace);
        nReplace = pNewListEnd - vResynIdSeq;
        
        iter++;
        printf("iter %d: number of subgraphs remained: %d\n", iter, nReplace);
    }

    // debug
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nResyn, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            checkInsertion(vResynRoots, vOldRoot2NewRootLits, vSelectedSubgInd,
                           nResyn);
        });
    q_ct1.wait();

    printf("Insertion complete, idCounter = %d\n", idCounter);

    sycl::free(vFinishedMark, q_ct1);
    sycl::free(vResynIdSeq, q_ct1);
    return idCounter;
}

inline int isRedundantNode(int nodeId, int nPIs, const int * fanin0) {
    return nodeId > nPIs && fanin0[nodeId] == dUtils::AigConst1;
}

int topoSortGetLevel(int nodeId, int nPIs, int * levels, const int * fanin0, const int * fanin1) {
    // printf("  Topo sorting nodeId %d ...\n", nodeId);
    assert(nodeId <= nPIs || fanin0[nodeId] != -1);

    if (levels[nodeId] != -1)
        return levels[nodeId];
    if (isRedundantNode(nodeId, nPIs, fanin0))
        return (levels[nodeId] = 
                topoSortGetLevel(AigNodeID(fanin1[nodeId]), nPIs, levels, fanin0, fanin1));
    return (levels[nodeId] =
                1 + std::max(topoSortGetLevel(AigNodeID(fanin0[nodeId]), nPIs,
                                              levels, fanin0, fanin1),
                             topoSortGetLevel(AigNodeID(fanin1[nodeId]), nPIs,
                                              levels, fanin0, fanin1)));
}

std::tuple<int, int *, int *, int *, int> reorder(int *vFanin0New,
                                                  int *vFanin1New, int *pOuts,
                                                  int nPIs, int nPOs, int nObjs,
                                                  int nBufferLen) {
    sycl::queue &q_ct1 = nsycl::queue();

    int * vhFanin0, * vhFanin1, * vhLevels, * vhNewInd;
    int * vhFanin0New, * vhFanin1New, * vhOutsNew;

    int nNodesNew, nObjsNew;

    // copy fanin arrays to host
    vhFanin0 = sycl::malloc_host<int>(nBufferLen, q_ct1);
    vhFanin1 = sycl::malloc_host<int>(nBufferLen, q_ct1);
    vhLevels = sycl::malloc_host<int>(nBufferLen, q_ct1);
    vhNewInd = sycl::malloc_host<int>(nBufferLen, q_ct1);

    q_ct1.memcpy(vhFanin0, vFanin0New, nBufferLen * sizeof(int)).wait();
    q_ct1.memcpy(vhFanin1, vFanin1New, nBufferLen * sizeof(int)).wait();
    memset(vhLevels, -1, nBufferLen * sizeof(int));

    printf("Start reordering ...\n");
    auto cpuSequentialStartTime = hrClock();

    // topo order to get level of each node. the redundant node does not contribute to level
    for (int i = 0; i <= nPIs; i++)
        vhLevels[i] = 0;
    for (int i = nPIs + 1; i < nBufferLen; i++)
        if (vhFanin0[i] != -1) {
            topoSortGetLevel(i, nPIs, vhLevels, vhFanin0, vhFanin1);
        }
    
    // count total number of nodes and assign each node an id level by level
    int nMaxLevel = 0;
    std::vector<int> vLevelNodesCount(1, 0);
    for (int i = nPIs + 1; i < nBufferLen; i++)
        if (vhFanin0[i] != -1 && !isRedundantNode(i, nPIs, vhFanin0)) {
            assert(vhLevels[i] > 0);
            if (vhLevels[i] > nMaxLevel) {
                while (vLevelNodesCount.size() < vhLevels[i] + 1)
                    vLevelNodesCount.push_back(0);
                nMaxLevel = vhLevels[i];
            }
            assert(vhLevels[i] < vLevelNodesCount.size());
            vLevelNodesCount[vhLevels[i]]++;
        }
    assert(vLevelNodesCount[0] == 0);
    
    for (int i = 1; i <= nMaxLevel; i++)
        vLevelNodesCount[i] += vLevelNodesCount[i - 1];
    nNodesNew = vLevelNodesCount.back();
    nObjsNew = nNodesNew + nPIs + 1;
    
    // assign consecutive new ids
    for (int i = nBufferLen - 1; i > nPIs; i--)
        if (vhFanin0[i] != -1 && !isRedundantNode(i, nPIs, vhFanin0))
            vhNewInd[i] = (--vLevelNodesCount[vhLevels[i]]) + nPIs + 1;
    // ids for PIs do not change
    for (int i = 0; i <= nPIs; i++)
        vhNewInd[i] = i;


    // gather nodes in assigned order
    vhFanin0New = (int *) malloc(nObjsNew * sizeof(int));
    vhFanin1New = (int *) malloc(nObjsNew * sizeof(int));
    vhOutsNew = (int *) malloc(nPOs * sizeof(int));
    memset(vhFanin0New, -1, nObjsNew * sizeof(int));
    memset(vhFanin1New, -1, nObjsNew * sizeof(int));

    for (int i = nPIs + 1; i < nBufferLen; i++)
        if (vhFanin0[i] != -1 && !isRedundantNode(i, nPIs, vhFanin0)) {
            assert(vhFanin0New[vhNewInd[i]] == -1 && vhFanin1New[vhNewInd[i]] == -1);
            // propagate if fanin is redundant
            int lit, propLit = vhFanin0[i];
            while(isRedundantNode(AigNodeID(propLit), nPIs, vhFanin0))
                propLit = dUtils::AigNodeNotCond(vhFanin1[AigNodeID(propLit)], AigNodeIsComplement(propLit));
            lit = dUtils::AigNodeLitCond(vhNewInd[AigNodeID(propLit)], AigNodeIsComplement(propLit));

            vhFanin0New[vhNewInd[i]] = lit;

            propLit = vhFanin1[i];
            while(isRedundantNode(AigNodeID(propLit), nPIs, vhFanin0))
                propLit = dUtils::AigNodeNotCond(vhFanin1[AigNodeID(propLit)], AigNodeIsComplement(propLit));
            lit = dUtils::AigNodeLitCond(vhNewInd[AigNodeID(propLit)], AigNodeIsComplement(propLit));
            vhFanin1New[vhNewInd[i]] = lit;

            if (vhFanin0New[vhNewInd[i]] > vhFanin1New[vhNewInd[i]]) {
                int temp = vhFanin0New[vhNewInd[i]];
                vhFanin0New[vhNewInd[i]] = vhFanin1New[vhNewInd[i]];
                vhFanin1New[vhNewInd[i]] = temp;
            }
        }
    
    // update POs
    for (int i = 0; i < nPOs; i++) {
        int oldId = AigNodeID(pOuts[i]);
        int lit, propLit;
        assert(oldId <= nPIs || vhFanin0[oldId] != -1);

        propLit = pOuts[i];
        while(isRedundantNode(AigNodeID(propLit), nPIs, vhFanin0))
            propLit = dUtils::AigNodeNotCond(vhFanin1[AigNodeID(propLit)], AigNodeIsComplement(propLit));
        lit = dUtils::AigNodeLitCond(vhNewInd[AigNodeID(propLit)], AigNodeIsComplement(propLit));

        vhOutsNew[i] = lit;
    }
    printf("Reordered network new nObjs: %d, original nObjs: %d\n", nObjsNew, nObjs);
    printf("Reordering complete!\n");
    printf(" ** CPU sequential time: %.2lf sec\n", (hrClock() - cpuSequentialStartTime) / (double) NS_PER_SEC);

    sycl::free(vhFanin0, q_ct1);
    sycl::free(vhFanin1, q_ct1);
    sycl::free(vhLevels, q_ct1);
    sycl::free(vhNewInd, q_ct1);

    return {nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nMaxLevel};
}

std::tuple<int, int *, int *, int *, int>
refactorMFFCPerform(bool fUseZeros, int cutSize, int nObjs, int nPIs, int nPOs,
                    int nNodes, const int *d_pFanin0, const int *d_pFanin1,
                    const int *d_pOuts, const int *d_pNumFanouts,
                    const int *d_pLevel, int *pOuts, int *pNumFanouts) {
    sycl::queue &q_ct1 = nsycl::queue();
    int * vRoots, * vNodesStatus;
    int * vNodesIndices;
    int * vCutTable, * vCutSizes, * vNumSaved;
    int * vResynRoots;
    int * vCuts, * vCutRanges;
    int * vTruthRanges;
    unsigned * vTruths, * vTruthsNeg, * vTruthElem;
    int * vNode2ConeResynIdx;
    uint64 * vSubgTable;
    int * vSubgLinks, * vSubgLens, * pSubgTableNext;
    int * vSelectedSubgInd;
    int * vOldRoot2NewRootLits;
    uint64 * vReconstructedKeys;
    uint32 * vReconstructedIds;
    int * vFanin0New, * vFanin1New;
    
    int * pNewGlobalListEnd;
    int nResyn, nCutArrayLen, nTruthArrayLen;
    int currLen;

    vRoots = sycl::malloc_device<int>(nObjs, q_ct1);
    vNodesStatus = sycl::malloc_device<int>(nObjs, q_ct1);
    vNodesIndices = sycl::malloc_device<int>(nObjs, q_ct1);
    vResynRoots = sycl::malloc_device<int>(nObjs, q_ct1);

    vCutTable = sycl::malloc_device<int>((size_t)nObjs * CUT_TABLE_SIZE, q_ct1);
    vCutSizes = sycl::malloc_device<int>(nObjs, q_ct1);
    vNumSaved = sycl::malloc_device<int>(nObjs, q_ct1);
    q_ct1.memset(vCutSizes, -1, nObjs * sizeof(int)).wait();
    q_ct1.memset(vNumSaved, -1, nObjs * sizeof(int)).wait();

    // precompute a consecutive indices array for gathering uses
    nsycl::iota(q_ct1, vNodesIndices,
               vNodesIndices + nObjs);

    // generate the initial vRoots
    pNewGlobalListEnd =
        std::copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), d_pOuts,
                     d_pOuts + nPOs, vRoots, dUtils::isNodeLit<int>(nPIs));
    currLen = pNewGlobalListEnd - vRoots;
    if (currLen == 0)
        return {-1, NULL, NULL, NULL, -1};
    printf("Gathered %d POs\n", currLen);
    std::transform(oneapi::dpl::execution::make_device_policy(q_ct1), vRoots,
                   pNewGlobalListEnd, vRoots, dUtils::getNodeID());
    // deduplicate
    oneapi::dpl::sort(oneapi::dpl::execution::make_device_policy(q_ct1), vRoots,
                      pNewGlobalListEnd);
    pNewGlobalListEnd =
        std::unique(oneapi::dpl::execution::make_device_policy(q_ct1), vRoots,
                    pNewGlobalListEnd);
    currLen = pNewGlobalListEnd - vRoots;

    int levelCount = 0;
    printf("Level %d, global list len %d\n", levelCount, currLen);

    while (currLen > 0) {
        q_ct1.memset(vNodesStatus, 0, nObjs * sizeof(int)).wait();

        q_ct1.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(currLen, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                recordMFFC<false>(vRoots, d_pFanin0, d_pFanin1, d_pNumFanouts,
                                  d_pLevel, vCutTable, vCutSizes, vNumSaved,
                                  nPIs, cutSize, currLen);
            });
        q_ct1.wait();

        q_ct1.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(currLen, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                setStatus(vRoots, vCutTable, vCutSizes, vNodesStatus, nPIs,
                          currLen);
            });
        q_ct1.wait();

        pNewGlobalListEnd = nsycl::copy_if(q_ct1, vNodesIndices,
            vNodesIndices + nObjs, vNodesStatus, vRoots, dUtils::isOne<int>());
        currLen = pNewGlobalListEnd - vRoots;
        
        levelCount++;
        printf("Level %d, global list len %d\n", levelCount, currLen);
    }
    sycl::free(vRoots, q_ct1);
    sycl::free(vNodesStatus, q_ct1);

    // filter out too small MFFCs by replacing cut size with -1
    nsycl::replace_if(q_ct1,
                     vCutSizes, vCutSizes + nObjs, vNumSaved, isSmallMFFC(),
                     -1);

    // printMffcCut<<<1, 1>>>(vCutTable, vCutSizes, vNumSaved, d_pFanin0, d_pFanin1, nNodes, nPIs, nPOs);
    // cudaDeviceSynchronize();

    // collect the number of cones to be resyned
    pNewGlobalListEnd = nsycl::copy_if(q_ct1,
        vNodesIndices + nPIs + 1, vNodesIndices + nObjs, vCutSizes + nPIs + 1,
        vResynRoots, dUtils::notEqualsVal<int, -1>());
    nResyn = pNewGlobalListEnd - vResynRoots;
    printf("Total number of cones to be resyned: %d\n", nResyn);
    if (nResyn == 0) {
        sycl::free(vCutTable, q_ct1);
        sycl::free(vCutSizes, q_ct1);
        sycl::free(vNumSaved, q_ct1);
        sycl::free(vNodesIndices, q_ct1);
        sycl::free(vResynRoots, q_ct1);
        return {-1, NULL, NULL, NULL, -1};
    }

    vCutRanges = sycl::malloc_device<int>(nResyn, q_ct1);
    vTruthRanges = sycl::malloc_device<int>(nResyn, q_ct1);

    // (optional) sort vResynRoots according to cut sizes
    // this is to make sure that consecutive threads has similar cut and cone sizes
    pNewGlobalListEnd = std::copy_if(
        oneapi::dpl::execution::make_device_policy(q_ct1), vCutSizes + nPIs + 1,
        vCutSizes + nObjs, vCutRanges, dUtils::notEqualsVal<int, -1>());
    assert(pNewGlobalListEnd - vCutRanges == nResyn);
    oneapi::dpl::sort_by_key(oneapi::dpl::execution::make_device_policy(q_ct1), vCutRanges,
               vCutRanges + nResyn, vResynRoots);

    // gather the cuts to be resyned into a consecutive array
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nResyn, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            getCutTruthRanges(vResynRoots, vCutSizes, vCutRanges, vTruthRanges,
                              nResyn);
        });
    q_ct1.wait();
    oneapi::dpl::inclusive_scan(
        oneapi::dpl::execution::make_device_policy(q_ct1), vCutRanges,
        vCutRanges + nResyn, vCutRanges);
    oneapi::dpl::inclusive_scan(
        oneapi::dpl::execution::make_device_policy(q_ct1), vTruthRanges,
        vTruthRanges + nResyn, vTruthRanges);
    q_ct1.wait();

    q_ct1.memcpy(&nCutArrayLen, &vCutRanges[nResyn - 1], sizeof(int));
    q_ct1.memcpy(&nTruthArrayLen, &vTruthRanges[nResyn - 1], sizeof(int))
        .wait();
    q_ct1.wait();
    vCuts = sycl::malloc_device<int>(nCutArrayLen, q_ct1);
    vTruths = sycl::malloc_device<unsigned int>(nTruthArrayLen, q_ct1);
    vTruthsNeg = sycl::malloc_device<unsigned int>(nTruthArrayLen, q_ct1);
    
        vTruthElem = sycl::malloc_device<unsigned int>(
            (size_t)cutSize * dUtils::TruthWordNum(cutSize), q_ct1);

    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nResyn, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            Table::gatherTableToConsecutive<int, CUT_TABLE_SIZE>(
                vCutTable, vCutSizes, vResynRoots, vCutRanges, vCuts, nResyn);
        });
    q_ct1.wait();
    sycl::free(vCutTable, q_ct1);

    // gather truth table and mark the MFFC nodes (i.e. to be deleted)
    vNode2ConeResynIdx = sycl::malloc_device<int>(nObjs, q_ct1);
    q_ct1.memset(vNode2ConeResynIdx, -1, nObjs * sizeof(int)).wait();

    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            Aig::getElemTruthTable(vTruthElem, cutSize);
        });
    q_ct1.wait();
    auto startTruthTime = hrClock();
    // Preallocated per-thread global scratch for intermediate truth tables
    // (replaces device-side malloc, which the Level-Zero backend rejects).
    int nWordsElemTT = (cutSize <= 5) ? 1 : (1 << (cutSize - 5));
    int maxPerThreadTT = STACK_SIZE * nWordsElemTT;
    size_t nThreadsTT =
        (size_t)NUM_BLOCKS(nResyn, THREAD_PER_BLOCK) * THREAD_PER_BLOCK;
    unsigned *vTruthMemScratch =
        sycl::malloc_device<unsigned>(nThreadsTT * maxPerThreadTT, q_ct1);
    // No subgroup collectives here, so no required subgroup size is needed;
    // the kernel runs with whatever size the device provides (32 or 64).
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nResyn, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            Aig::getCutTruthTableConsecutive<STACK_SIZE>(
                d_pFanin0, d_pFanin1, vNumSaved, vResynRoots, vCuts,
                vCutRanges, vTruths, vTruthRanges, vTruthElem, nResyn, nPIs,
                cutSize, vTruthMemScratch, maxPerThreadTT,
                vNode2ConeResynIdx);
        });
    q_ct1.wait();
    sycl::free(vTruthMemScratch, q_ct1);
    std::transform(oneapi::dpl::execution::make_device_policy(q_ct1), vTruths,
                   vTruths + nTruthArrayLen, vTruthsNeg,
                   bitwiseNot()); // get the negated truth table
    q_ct1.wait();
    printf("Truth table computation time: %.2lf sec\n", (hrClock() - startTruthTime) / (double) NS_PER_SEC);

    // resynthesize cones
    
    // vSubgLinks indicating idx of next row, if one row in vSubgTable is not enough:
    // -1: unvisited, 0: last row, >0: next row idx
    int nResynGraphs = 2 * nResyn; // for normal and negated graphs
    vSelectedSubgInd = sycl::malloc_device<int>(nResyn, q_ct1);
    vSubgTable = sycl::malloc_device<uint64>(
        (size_t)2 * nResynGraphs * SUBG_TABLE_SIZE, q_ct1);
    vSubgLinks = sycl::malloc_device<int>((size_t)2 * nResynGraphs, q_ct1);
    vSubgLens = sycl::malloc_device<int>((size_t)nResynGraphs, q_ct1);
    pSubgTableNext = sycl::malloc_device<int>(1, q_ct1);
    q_ct1.memset(vSubgLinks, -1, (size_t)2 * nResynGraphs * sizeof(int)).wait();
    q_ct1.memset(vSubgLens, -1, (size_t)nResynGraphs * sizeof(int)).wait();
    q_ct1.memcpy(pSubgTableNext, &nResynGraphs, sizeof(int)).wait();
    q_ct1.wait();

    auto startResynTime = hrClock();
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nResynGraphs, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            factorFromTruth(vCuts, vCutRanges, vSubgTable, vSubgLinks,
                            vSubgLens, pSubgTableNext, vTruths, vTruthsNeg,
                            vTruthRanges, vTruthElem, nResyn);
        });
    q_ct1.wait();
    printf("ISOP + factor time: %.2lf sec\n", (hrClock() - startResynTime) / (double) NS_PER_SEC);

    // create hashtable and evaluate number of added nodes for each cone
    HashTable<uint64, uint32> hashTable((int)(nObjs / (HT_LOAD_FACTOR * 1.5)));
    uint64 * htKeys = hashTable.get_keys_storage();
    uint32 * htValues = hashTable.get_values_storage();
    int htCapacity = hashTable.get_capacity();

    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nNodes, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            Aig::buildHashTable(d_pFanin0, d_pFanin1, htKeys, htValues,
                                htCapacity, nNodes, nPIs);
        });
    q_ct1.wait();

    // the evaluation is DAG-aware, considering shareable node with non-MFFC nodes (vNode2ConeResynIdx unassigned)
    // The subgroup XOR shuffle pairs adjacent lanes (delta 1), which is valid
    // for any even subgroup size, so this single kernel is portable across
    // 32- and 64-wide subgroups without forcing a required size.
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nResynGraphs, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            evalFactoredForm(vResynRoots, vCuts, vCutRanges, vNumSaved,
                             d_pLevel, vNode2ConeResynIdx, htKeys, htValues,
                             htCapacity, vSubgTable, vSubgLinks, vSubgLens,
                             vSelectedSubgInd, nResyn);
        });
    q_ct1.wait();
    // vSelectedSubgInd is now aligned with vResynRoots

    // create a new hashtable containing all the non-MFFC nodes and root nodes with same ids
    // i.e., the MFFC nodes except roots are removed
    HashTable<uint64, uint32> hashTableNew((int)(nObjs / HT_LOAD_FACTOR));
    uint64 * htNewKeys = hashTableNew.get_keys_storage();
    uint32 * htNewValues = hashTableNew.get_values_storage();
    int htNewCapacity = hashTableNew.get_capacity();
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(htCapacity, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            duplicateHashTableWithoutMFFCs(
                vNode2ConeResynIdx, vSelectedSubgInd, htKeys, htValues,
                htCapacity, htNewKeys, htNewValues, htNewCapacity);
        });
    q_ct1.wait();
    hashTable.freeMem();

    // insert subgraphs into the new hashtable
    vOldRoot2NewRootLits = sycl::malloc_device<int>(nObjs, q_ct1);
    q_ct1.memset(vOldRoot2NewRootLits, -1, nObjs * sizeof(int)).wait();
    int nBufferLen = insertMFFCs(htNewKeys, htNewValues, htNewCapacity, vSubgTable, vSubgLinks, vSubgLens,
                                 vResynRoots, vCuts, vCutRanges, vSelectedSubgInd, vOldRoot2NewRootLits, nObjs, nResyn);
    q_ct1.wait();

    // dump all entries from the hashtable
    vReconstructedKeys = sycl::malloc_device<uint64>(nBufferLen, q_ct1);
    vReconstructedIds = sycl::malloc_device<uint32>(nBufferLen, q_ct1);
    vFanin0New = sycl::malloc_device<int>(nBufferLen, q_ct1);
    vFanin1New = sycl::malloc_device<int>(nBufferLen, q_ct1);
    q_ct1.memset(vFanin0New, -1, nBufferLen * sizeof(int)).wait();
    q_ct1.memset(vFanin1New, -1, nBufferLen * sizeof(int)).wait();
    int nEntries = hashTableNew.retrieve_all(vReconstructedKeys, vReconstructedIds, nBufferLen, 1);
    q_ct1.wait();
    hashTableNew.freeMem();

    // unbind keys to fanin arrays, and modify the original root nodes to be buffers of the new roots
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nEntries, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            unbindKeysUpdateOldRoots(vOldRoot2NewRootLits, vReconstructedKeys,
                                     vReconstructedIds, vFanin0New, vFanin1New,
                                     nEntries, nObjs, nBufferLen);
        });
    q_ct1.wait();

    auto reorderStartTime = hrClock();
    auto [nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nLevelsNew] = reorder(
        vFanin0New, vFanin1New, pOuts, nPIs, nPOs, nObjs, nBufferLen
    );
    printf("Sequential reorder time: %.2lf secs\n", (hrClock() - reorderStartTime) / (double) NS_PER_SEC);

    sycl::free(vNodesIndices, q_ct1);
    sycl::free(vCutSizes, q_ct1);
    sycl::free(vNumSaved, q_ct1);
    sycl::free(vResynRoots, q_ct1);
    sycl::free(vCuts, q_ct1);
    sycl::free(vCutRanges, q_ct1);
    sycl::free(vTruthRanges, q_ct1);
    sycl::free(vTruths, q_ct1);
    sycl::free(vTruthsNeg, q_ct1);
    sycl::free(vTruthElem, q_ct1);
    sycl::free(vNode2ConeResynIdx, q_ct1);
    sycl::free(vSubgTable, q_ct1);
    sycl::free(vSubgLinks, q_ct1);
    sycl::free(vSubgLens, q_ct1);
    sycl::free(pSubgTableNext, q_ct1);
    sycl::free(vSelectedSubgInd, q_ct1);
    sycl::free(vOldRoot2NewRootLits, q_ct1);
    sycl::free(vReconstructedKeys, q_ct1);
    sycl::free(vReconstructedIds, q_ct1);
    sycl::free(vFanin0New, q_ct1);
    sycl::free(vFanin1New, q_ct1);

    return {nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nLevelsNew};
}
