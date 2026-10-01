#include <tuple>
#include <vector>
#include <ctime>
#include <climits>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include "refactor.h"
#include "mffc.cuh"
#include "truth.cuh"
#include "strash.cuh"
#include "tables.cuh"
#include "common.h"
#include "alg_factor.cuh"

using namespace sop;

struct isSmallMFFC {
    __host__ __device__
    bool operator()(const int elem) const {
        return elem != -1 && elem < 2;
    }
};

struct bitwiseNot {
    __host__ __device__
    unsigned operator()(const unsigned elem) const {
        return ~elem;
    }
};

struct identity {
    template <typename T>
    __host__ __device__ T operator()(const T& x) const { return x; }
};


// debug functions
__global__ void printMffcCut(int * vCutTable, int * vCutSizes, int * vConeSizes,
                             const int * pFanin0, const int * pFanin1, 
                             int nNodes, int nPIs, int nPOs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0)
        return;

    int counter = 0;
    for (int i = 0; i < nNodes; i++) {
        int id = i + nPIs + 1;
        if (vCutSizes[id] == -1)
            continue;
        counter++;
        
        printf("root: %d, cone size: %d | ", id, vConeSizes[id]);
        for (int j = 0; j < vCutSizes[id]; j++) {
            printf("%d ", vCutTable[id * CUT_TABLE_SIZE + j]);
        }
        printf("\n");
    }
    printf("Total number of MFFCs: %d\n", counter);
}


template <bool useHashtable = false>
__global__ void recordMFFC(const int * vRoots, 
                           const int * pFanin0, const int * pFanin1, 
                           const int * pNumFanouts, const int * pLevels, 
                           int * vCutTable, int * vCutSizes, int * vConeSizes, 
                           int nPIs, int nMaxCutSize, int nRoots) {
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void setStatus(const int * vRoots,
                          const int * vCutTable, const int * vCutSizes,
                          int * vNodesStatus,
                          int nPIs, int nRoots) {
    // vNodesStatus should be set to all zeros before launching this kernel
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void getCutTruthRanges(const int * vResynRoots, const int * vCutSizes, 
                                  int * vCutRanges, int * vTruthRanges, int nResyn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nResyn) {
        int nodeId = vResynRoots[idx];
        int cutSize = vCutSizes[nodeId];
        assert(cutSize > 0);
        vCutRanges[idx] = cutSize;
        vTruthRanges[idx] = dUtils::TruthWordNum(cutSize);
    }
}

__device__ int evalSubgNumAdded(int rootId, int * pNewRootLevel, const int * vCurrCut, int nVars, 
                                int nNodeMax, int nLevelMax, const int * pLevels, const int * vNode2ConeResynIdx,
                                const uint64 * htKeys, const uint32 * htValues, int htCapacity,
                                const uint64 * vSubgTable, const int * vSubgLinks, const int * vSubgLens, 
                                int subgIdx) {
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
        newLevel = 1 + max(vLevels[id0], vLevels[id1]);
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

__global__ void evalFactoredForm(const int * vResynRoots, const int * vCuts, const int * vCutRanges,
                                 const int * vNumSaved, const int * pLevels, const int * vNode2ConeResynIdx, 
                                 const uint64 * htKeys, const uint32 * htValues, int htCapacity,
                                 const uint64 * vSubgTable, const int * vSubgLinks, const int * vSubgLens, 
                                 int * vSelectedSubgInd, int nResyn) {
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int coneIdx, rootId;
    int nVars, nSaved, nAdded, nOtherAdded, nNewLevel, nOtherNewLevel;
    int startIdx, endIdx;
    unsigned warpMask;
    int fSelectedSubg;

    for (; idx < 2 * nResyn; idx += nThreads) {
        warpMask = __activemask();
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
        nOtherAdded = __shfl_xor_sync(warpMask, nAdded, 1); // nAdded of the corresponding negated subgraph
        nOtherNewLevel = __shfl_xor_sync(warpMask, nNewLevel, 1);
        
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

__global__ void duplicateHashTableWithoutMFFCs(const int * vNode2ConeResynIdx, const int * vSelectedSubgInd,
                                               const uint64 * htKeys, const uint32 * htValues, int htCapacity,
                                               uint64 * htDestKeys, uint32 * htDestValues, int htDestCapacity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void checkSingleVarSubg(const uint64 * vSubgTable, const int * vSubgLinks, const int * vSubgLens,
                                   const int * vResynRoots, const int * vSelectedSubgInd, int * vOldRoot2NewRootLits, 
                                   int * vFinishedMark, int nObjs, int nResyn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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


__global__ void insertSubgIter(int iter, const int * vResynIdSeq,
                               const int * vCuts, const int * vCutRanges,
                               uint64 * htDestKeys, uint32 * htDestValues, int htDestCapacity,
                               uint64 * vSubgTable, const int * vSubgLinks, const int * vSubgLens,
                               const int * vSelectedSubgInd, int idCounter, int nReplace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void updateInsertedIdsIter(int iter, const int * vResynRoots, const int * vResynIdSeq,
                                      const uint64 * htDestKeys, const uint32 * htDestValues, int htDestCapacity,
                                      uint64 * vSubgTable, const int * vSubgLinks, const int * vSubgLens,
                                      const int * vSelectedSubgInd, int * vOldRoot2NewRootLits, int * vFinishedMark, 
                                      int nReplace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void checkInsertion(const int * vResynRoots, const int * vOldRoot2NewRootLits, 
                               const int * vSelectedSubgInd, int nResyn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nResyn) {
        int subgIdx = vSelectedSubgInd[idx];
        int rootId = vResynRoots[idx];
        if (subgIdx != -1) {
            assert(vOldRoot2NewRootLits[rootId] != -1);
        }
    }
}

__global__ void unbindKeysUpdateOldRoots(const int * vOldRoot2NewRootLits, 
                                         const uint64 * vReconstructedKeys, const uint32 * vReconstructedIds,
                                         int * vFanin0New, int * vFanin1New, int nEntries, int nObjs, int nBufferLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

int insertMFFCs(uint64 * htDestKeys, uint32 * htDestValues, int htDestCapacity,
                uint64 * vSubgTable, int * vSubgLinks, int * vSubgLens, 
                const int * vResynRoots, const int * vCuts, const int * vCutRanges,
                const int * vSelectedSubgInd, int * vOldRoot2NewRootLits, 
                int nObjs, int nResyn) {
    // create a sequence of resyn indices and shrink it iteratively,
    // but do not change vSelectedSubgInd since it is aligned with vResynRoots
    int * vResynIdSeq, * vFinishedMark, * pNewListEnd;
    cudaMalloc(&vResynIdSeq, nResyn * sizeof(int));
    cudaMalloc(&vFinishedMark, nResyn * sizeof(int));
    cudaMemset(vFinishedMark, 0, nResyn * sizeof(int));
    thrust::sequence(thrust::device, vResynIdSeq, vResynIdSeq + nResyn);

    // process the single var subgraphs first
    checkSingleVarSubg<<<NUM_BLOCKS(nResyn, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vSubgTable, vSubgLinks, vSubgLens, vResynRoots, vSelectedSubgInd, vOldRoot2NewRootLits, 
        vFinishedMark, nObjs, nResyn
    );
    cudaDeviceSynchronize();
    pNewListEnd = thrust::remove_if(thrust::device, vResynIdSeq, vResynIdSeq + nResyn, 
                                    vFinishedMark, identity{});
    assert(pNewListEnd - vResynIdSeq <= nResyn);
    int nReplace = pNewListEnd - vResynIdSeq;
    printf("Number of subgraphs to be inserted: %d\n", nReplace);

    int iter = 0; // the index of subgraph nodes that are being processed in the current iteration
    int idCounter = nObjs; // used for assigning new tentative ids of inserted nodes
    while (nReplace > 0) {
        cudaMemset(vFinishedMark, 0, nReplace * sizeof(int));
        insertSubgIter<<<NUM_BLOCKS(nReplace, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            iter, vResynIdSeq, vCuts, vCutRanges, htDestKeys, htDestValues, htDestCapacity, 
            vSubgTable, vSubgLinks, vSubgLens, vSelectedSubgInd, idCounter, nReplace
        );
        cudaDeviceSynchronize();
        updateInsertedIdsIter<<<NUM_BLOCKS(nReplace, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            iter, vResynRoots, vResynIdSeq, htDestKeys, htDestValues, htDestCapacity, 
            vSubgTable, vSubgLinks, vSubgLens, vSelectedSubgInd, vOldRoot2NewRootLits, 
            vFinishedMark, nReplace
        );
        cudaDeviceSynchronize();
        // increment idCounter
        assert(idCounter + nReplace < (INT_MAX / 2));
        idCounter += nReplace;

        // shrink according to vFinishedMark
        pNewListEnd = thrust::remove_if(thrust::device, vResynIdSeq, vResynIdSeq + nReplace, 
                                        vFinishedMark, identity{});
        cudaDeviceSynchronize();
        assert(pNewListEnd - vResynIdSeq <= nReplace);
        nReplace = pNewListEnd - vResynIdSeq;
        
        iter++;
        printf("iter %d: number of subgraphs remained: %d\n", iter, nReplace);
    }

    // debug
    checkInsertion<<<NUM_BLOCKS(nResyn, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vResynRoots, vOldRoot2NewRootLits, vSelectedSubgInd, nResyn
    );
    cudaDeviceSynchronize();

    printf("Insertion complete, idCounter = %d\n", idCounter);

    cudaFree(vFinishedMark);
    cudaFree(vResynIdSeq);
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
            1 + max(
                topoSortGetLevel(AigNodeID(fanin0[nodeId]), nPIs, levels, fanin0, fanin1),
                topoSortGetLevel(AigNodeID(fanin1[nodeId]), nPIs, levels, fanin0, fanin1)
            ));
}

std::tuple<int, int *, int *, int *, int>
reorder(int * vFanin0New, int * vFanin1New, int * pOuts,
        int nPIs, int nPOs, int nObjs, int nBufferLen) {
    
    int * vhFanin0, * vhFanin1, * vhLevels, * vhNewInd;
    int * vhFanin0New, * vhFanin1New, * vhOutsNew;

    int nNodesNew, nObjsNew;

    // copy fanin arrays to host
    cudaHostAlloc(&vhFanin0, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhFanin1, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhLevels, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhNewInd, nBufferLen * sizeof(int), cudaHostAllocDefault);

    cudaMemcpy(vhFanin0, vFanin0New, nBufferLen * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vhFanin1, vFanin1New, nBufferLen * sizeof(int), cudaMemcpyDeviceToHost);
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

    cudaFreeHost(vhFanin0);
    cudaFreeHost(vhFanin1);
    cudaFreeHost(vhLevels);
    cudaFreeHost(vhNewInd);

    return {nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nMaxLevel};
}


std::tuple<int, int *, int *, int *, int> 
refactorMFFCPerform(bool fUseZeros, int cutSize,
                    int nObjs, int nPIs, int nPOs, int nNodes, 
                    const int * d_pFanin0, const int * d_pFanin1, const int * d_pOuts, 
                    const int * d_pNumFanouts, const int * d_pLevel, 
                    int * pOuts, int * pNumFanouts) {
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

    cudaMalloc(&vRoots, nObjs * sizeof(int));
    cudaMalloc(&vNodesStatus, nObjs * sizeof(int));
    cudaMalloc(&vNodesIndices, nObjs * sizeof(int));
    cudaMalloc(&vResynRoots, nObjs * sizeof(int));

    cudaMalloc(&vCutTable, (size_t)nObjs * CUT_TABLE_SIZE * sizeof(int));
    cudaMalloc(&vCutSizes, nObjs * sizeof(int));
    cudaMalloc(&vNumSaved, nObjs * sizeof(int));
    cudaMemset(vCutSizes, -1, nObjs * sizeof(int));
    cudaMemset(vNumSaved, -1, nObjs * sizeof(int));

    // precompute a consecutive indices array for gathering uses
    thrust::sequence(thrust::device, vNodesIndices, vNodesIndices + nObjs);

    // generate the initial vRoots
    pNewGlobalListEnd = thrust::copy_if(thrust::device, d_pOuts, d_pOuts + nPOs, 
                                        vRoots, dUtils::isNodeLit<int>(nPIs));
    currLen = pNewGlobalListEnd - vRoots;
    if (currLen == 0)
        return {-1, NULL, NULL, NULL, -1};
    printf("Gathered %d POs\n", currLen);
    thrust::transform(thrust::device, vRoots, pNewGlobalListEnd, vRoots, dUtils::getNodeID());
    // deduplicate
    thrust::sort(thrust::device, vRoots, pNewGlobalListEnd);
    pNewGlobalListEnd = thrust::unique(thrust::device, vRoots, pNewGlobalListEnd);
    currLen = pNewGlobalListEnd - vRoots;

    int levelCount = 0;
    printf("Level %d, global list len %d\n", levelCount, currLen);

    while (currLen > 0) {
        cudaMemset(vNodesStatus, 0, nObjs * sizeof(int));

        recordMFFC<false><<<NUM_BLOCKS(currLen, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vRoots, d_pFanin0, d_pFanin1, d_pNumFanouts, d_pLevel, 
            vCutTable, vCutSizes, vNumSaved, 
            nPIs, cutSize, currLen
        );
        cudaDeviceSynchronize();

        setStatus<<<NUM_BLOCKS(currLen, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vRoots, vCutTable, vCutSizes, vNodesStatus, nPIs, currLen
        );
        cudaDeviceSynchronize();

        pNewGlobalListEnd = thrust::copy_if(
            thrust::device, vNodesIndices, vNodesIndices + nObjs, 
            vNodesStatus, vRoots, dUtils::isOne<int>()
        );
        currLen = pNewGlobalListEnd - vRoots;
        
        levelCount++;
        printf("Level %d, global list len %d\n", levelCount, currLen);
    }
    cudaFree(vRoots);
    cudaFree(vNodesStatus);

    // filter out too small MFFCs by replacing cut size with -1
    thrust::replace_if(thrust::device, vCutSizes, vCutSizes + nObjs, vNumSaved, isSmallMFFC(), -1);

    // printMffcCut<<<1, 1>>>(vCutTable, vCutSizes, vNumSaved, d_pFanin0, d_pFanin1, nNodes, nPIs, nPOs);
    // cudaDeviceSynchronize();

    // collect the number of cones to be resyned
    pNewGlobalListEnd = thrust::copy_if(
        thrust::device, vNodesIndices + nPIs + 1, vNodesIndices + nObjs, 
        vCutSizes + nPIs + 1, vResynRoots, dUtils::notEqualsVal<int, -1>()
    );
    nResyn = pNewGlobalListEnd - vResynRoots;
    printf("Total number of cones to be resyned: %d\n", nResyn);
    if (nResyn == 0) {
        cudaFree(vCutTable);
        cudaFree(vCutSizes);
        cudaFree(vNumSaved);
        cudaFree(vNodesIndices);
        cudaFree(vResynRoots);
        return {-1, NULL, NULL, NULL, -1};
    }

    cudaMalloc(&vCutRanges, nResyn * sizeof(int));
    cudaMalloc(&vTruthRanges, nResyn * sizeof(int));

    // (optional) sort vResynRoots according to cut sizes
    // this is to make sure that consecutive threads has similar cut and cone sizes
    pNewGlobalListEnd = thrust::copy_if(
        thrust::device, vCutSizes + nPIs + 1, vCutSizes + nObjs, 
        vCutRanges, dUtils::notEqualsVal<int, -1>()
    );
    assert(pNewGlobalListEnd - vCutRanges == nResyn);
    thrust::sort_by_key(thrust::device, vCutRanges, vCutRanges + nResyn, vResynRoots);

    // gather the cuts to be resyned into a consecutive array
    getCutTruthRanges<<<NUM_BLOCKS(nResyn, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vResynRoots, vCutSizes, vCutRanges, vTruthRanges, nResyn);
    cudaDeviceSynchronize();
    thrust::inclusive_scan(thrust::device, vCutRanges, vCutRanges + nResyn, vCutRanges);
    thrust::inclusive_scan(thrust::device, vTruthRanges, vTruthRanges + nResyn, vTruthRanges);
    cudaDeviceSynchronize();

    cudaMemcpy(&nCutArrayLen, &vCutRanges[nResyn - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&nTruthArrayLen, &vTruthRanges[nResyn - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMalloc(&vCuts, nCutArrayLen * sizeof(int));
    cudaMalloc(&vTruths, nTruthArrayLen * sizeof(unsigned));
    cudaMalloc(&vTruthsNeg, nTruthArrayLen * sizeof(unsigned));
    gpuErrchk( cudaMalloc(&vTruthElem, (size_t)cutSize * dUtils::TruthWordNum(cutSize) * sizeof(unsigned)) );

    Table::gatherTableToConsecutive<int, CUT_TABLE_SIZE>
    <<<NUM_BLOCKS(nResyn, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vCutTable, vCutSizes, vResynRoots, vCutRanges, vCuts, nResyn
    );
    cudaDeviceSynchronize();
    cudaFree(vCutTable);

    // gather truth table and mark the MFFC nodes (i.e. to be deleted)
    cudaMalloc(&vNode2ConeResynIdx, nObjs * sizeof(int));
    cudaMemset(vNode2ConeResynIdx, -1, nObjs * sizeof(int));

    Aig::getElemTruthTable<<<1, 1>>>(vTruthElem, cutSize);
    cudaDeviceSynchronize();
    auto startTruthTime = hrClock();
    Aig::getCutTruthTableConsecutive<STACK_SIZE><<<NUM_BLOCKS(nResyn, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        d_pFanin0, d_pFanin1, vNumSaved, vResynRoots, vCuts, vCutRanges, 
        vTruths, vTruthRanges, vTruthElem, nResyn, nPIs, cutSize, vNode2ConeResynIdx
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    thrust::transform(thrust::device, vTruths, vTruths + nTruthArrayLen, vTruthsNeg, bitwiseNot()); // get the negated truth table
    cudaDeviceSynchronize();
    printf("Truth table computation time: %.2lf sec\n", (hrClock() - startTruthTime) / (double) NS_PER_SEC);

    // resynthesize cones
    
    // vSubgLinks indicating idx of next row, if one row in vSubgTable is not enough:
    // -1: unvisited, 0: last row, >0: next row idx
    int nResynGraphs = 2 * nResyn; // for normal and negated graphs
    cudaMalloc(&vSelectedSubgInd, nResyn * sizeof(int));
    cudaMalloc(&vSubgTable, (size_t)2 * nResynGraphs * SUBG_TABLE_SIZE * sizeof(uint64));
    cudaMalloc(&vSubgLinks, (size_t)2 * nResynGraphs * sizeof(int));
    cudaMalloc(&vSubgLens, (size_t)nResynGraphs * sizeof(int));
    cudaMalloc(&pSubgTableNext, sizeof(int));
    cudaMemset(vSubgLinks, -1, (size_t)2 * nResynGraphs * sizeof(int));
    cudaMemset(vSubgLens, -1, (size_t)nResynGraphs * sizeof(int));
    cudaMemcpy(pSubgTableNext, &nResynGraphs, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    auto startResynTime = hrClock();
    factorFromTruth<<<NUM_BLOCKS(nResynGraphs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vCuts, vCutRanges, vSubgTable, vSubgLinks, vSubgLens, pSubgTableNext,
        vTruths, vTruthsNeg, vTruthRanges, vTruthElem, nResyn
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("ISOP + factor time: %.2lf sec\n", (hrClock() - startResynTime) / (double) NS_PER_SEC);

    // create hashtable and evaluate number of added nodes for each cone
    HashTable<uint64, uint32> hashTable((int)(nObjs / (HT_LOAD_FACTOR * 1.5)));
    uint64 * htKeys = hashTable.get_keys_storage();
    uint32 * htValues = hashTable.get_values_storage();
    int htCapacity = hashTable.get_capacity();

    Aig::buildHashTable<<<NUM_BLOCKS(nNodes, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        d_pFanin0, d_pFanin1, htKeys, htValues, htCapacity, nNodes, nPIs);
    cudaDeviceSynchronize();
    
    // the evaluation is DAG-aware, considering shareable node with non-MFFC nodes (vNode2ConeResynIdx unassigned)
    evalFactoredForm<<<NUM_BLOCKS(nResynGraphs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
    // evalFactoredForm<<<1, 2>>>(
        vResynRoots, vCuts, vCutRanges, vNumSaved, d_pLevel, vNode2ConeResynIdx, 
        htKeys, htValues, htCapacity, vSubgTable, vSubgLinks, vSubgLens, vSelectedSubgInd, nResyn
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // vSelectedSubgInd is now aligned with vResynRoots

    // create a new hashtable containing all the non-MFFC nodes and root nodes with same ids
    // i.e., the MFFC nodes except roots are removed
    HashTable<uint64, uint32> hashTableNew((int)(nObjs / HT_LOAD_FACTOR));
    uint64 * htNewKeys = hashTableNew.get_keys_storage();
    uint32 * htNewValues = hashTableNew.get_values_storage();
    int htNewCapacity = hashTableNew.get_capacity();
    duplicateHashTableWithoutMFFCs<<<NUM_BLOCKS(htCapacity, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vNode2ConeResynIdx, vSelectedSubgInd, htKeys, htValues, htCapacity, htNewKeys, htNewValues, htNewCapacity
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    hashTable.freeMem();

    // insert subgraphs into the new hashtable
    cudaMalloc(&vOldRoot2NewRootLits, nObjs * sizeof(int));
    cudaMemset(vOldRoot2NewRootLits, -1, nObjs * sizeof(int));
    int nBufferLen = insertMFFCs(htNewKeys, htNewValues, htNewCapacity, vSubgTable, vSubgLinks, vSubgLens,
                                 vResynRoots, vCuts, vCutRanges, vSelectedSubgInd, vOldRoot2NewRootLits, nObjs, nResyn);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // dump all entries from the hashtable
    cudaMalloc(&vReconstructedKeys, nBufferLen * sizeof(uint64));
    cudaMalloc(&vReconstructedIds, nBufferLen * sizeof(uint32));
    cudaMalloc(&vFanin0New, nBufferLen * sizeof(int));
    cudaMalloc(&vFanin1New, nBufferLen * sizeof(int));
    cudaMemset(vFanin0New, -1, nBufferLen * sizeof(int));
    cudaMemset(vFanin1New, -1, nBufferLen * sizeof(int));
    int nEntries = hashTableNew.retrieve_all(vReconstructedKeys, vReconstructedIds, nBufferLen, 1);
    cudaDeviceSynchronize();
    hashTableNew.freeMem();

    // unbind keys to fanin arrays, and modify the original root nodes to be buffers of the new roots
    unbindKeysUpdateOldRoots<<<NUM_BLOCKS(nEntries, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vOldRoot2NewRootLits, vReconstructedKeys, vReconstructedIds, 
        vFanin0New, vFanin1New, nEntries, nObjs, nBufferLen
    );
    cudaDeviceSynchronize();

    auto reorderStartTime = hrClock();
    auto [nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nLevelsNew] = reorder(
        vFanin0New, vFanin1New, pOuts, nPIs, nPOs, nObjs, nBufferLen
    );
    printf("Sequential reorder time: %.2lf secs\n", (hrClock() - reorderStartTime) / (double) NS_PER_SEC);

    cudaFree(vNodesIndices);
    cudaFree(vCutSizes);
    cudaFree(vNumSaved);
    cudaFree(vResynRoots);
    cudaFree(vCuts);
    cudaFree(vCutRanges);
    cudaFree(vTruthRanges);
    cudaFree(vTruths);
    cudaFree(vTruthsNeg);
    cudaFree(vTruthElem);
    cudaFree(vNode2ConeResynIdx);
    cudaFree(vSubgTable);
    cudaFree(vSubgLinks);
    cudaFree(vSubgLens);
    cudaFree(pSubgTableNext);
    cudaFree(vSelectedSubgInd);
    cudaFree(vOldRoot2NewRootLits);
    cudaFree(vReconstructedKeys);
    cudaFree(vReconstructedIds);
    cudaFree(vFanin0New);
    cudaFree(vFanin1New);

    return {nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nLevelsNew};
}
