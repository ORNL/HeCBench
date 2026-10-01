#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <tuple>
#include <functional>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include "common.h"
#include "aig_manager.h"
#include "hash_table.h"
#include "refactor.h"
#include "truth.cuh"
#include "strash.cuh"
#include "sop.cuh"
#include "print.cuh"

__managed__ int nNewObjs;

struct isNotSmallMffc {
    __host__ __device__
    bool operator()(const thrust::tuple<int, int> &e) const {
        return thrust::get<0>(e) == -1 || thrust::get<1>(e) >= 2;
    }
};

__device__ int decrementRef(int rootId, int nodeId, int nPIs, int nMaxCutSize, int nMinLevel,
                            const int * pNumFanouts, const int * pLevels, 
                            int * vCutTable, int * vCutSizes, 
                            int * pTravSize, int * travIds, int * travRefs) {
    if (dUtils::AigIsPIConst(nodeId, nPIs) || pLevels[nodeId] <= nMinLevel) {
        // stop expansion, also do not add into the trav list
        int oldCutSize = vCutSizes[rootId];
        if (oldCutSize < nMaxCutSize) {
            // add into cut list if it is not inside
            for (int i = 0; i < oldCutSize; i++)
                if (vCutTable[rootId * CUT_TABLE_SIZE + i] == nodeId)
                    return 1;
            vCutTable[rootId * CUT_TABLE_SIZE + oldCutSize] = nodeId;
            vCutSizes[rootId]++;
            return 1;
        } else {
            // the cut has reached max size
            vCutSizes[rootId] = -1;
            return -100;
        }
    }

    // check whether nodeId is already in the trav list
    for (int i = 0; i < *pTravSize; i++)
        if (travIds[i] == nodeId)
            return --travRefs[i];
    assert(*pTravSize < STACK_SIZE);

    // nodeId is not in the trav list; insert it
    travIds[*pTravSize] = nodeId;
    travRefs[*pTravSize] = pNumFanouts[nodeId] - 1;
    (*pTravSize)++;
    return pNumFanouts[nodeId] - 1;
}

__global__ void getMffcCut(const int * pFanin0, const int * pFanin1, 
                           const int * pNumFanouts, const int * pLevels, 
                           int * vCutTable, int * vCutSizes, int * vConeSizes,
                           int nNodes, int nPIs, int nMaxCutSize) {
    // TODO for hyp, there are some nodes whose MFFC is not the same as ABC
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stack[STACK_SIZE], travIds[STACK_SIZE], travRefs[STACK_SIZE];
    int stackTop, travSize, coneSize;
    int nodeId, rootId, faninId, nMinLevel;
    int fDecRet;

    for (; idx < nNodes; idx += nThreads) {
        stackTop = -1, travSize = 0, coneSize = 0;

        rootId = idx + nPIs + 1;
        stack[++stackTop] = rootId; // do not launch threads for PIs
        nMinLevel = max(0, pLevels[rootId] - 10);

        // printf("rootId = %d\n", rootId);
        // printf("  minLv: %d\n", nMinLevel);

        while (stackTop != -1) {
            nodeId = stack[stackTop--];
            coneSize++;
            // printf("  %d \n", nodeId);

            // check its two fanins
            faninId = dUtils::AigNodeID(pFanin1[nodeId]);
            // printf("    checking fanin %d\n", faninId);
            fDecRet = decrementRef(rootId, faninId, nPIs, nMaxCutSize, nMinLevel, 
                                   pNumFanouts, pLevels, vCutTable, vCutSizes, 
                                   &travSize, travIds, travRefs);
            // printf("    checked fanin %d\n", faninId);
            if (fDecRet == -100)
                break;             // cut size reached maximum
            else if (fDecRet == 0)
                stack[++stackTop] = faninId;
            assert(stackTop < STACK_SIZE);

            faninId = dUtils::AigNodeID(pFanin0[nodeId]);
            // printf("    checking fanin %d\n", faninId);
            fDecRet = decrementRef(rootId, faninId, nPIs, nMaxCutSize, nMinLevel, 
                                   pNumFanouts, pLevels, vCutTable, vCutSizes, 
                                   &travSize, travIds, travRefs);
            // printf("    checked fanin %d\n", faninId);
            if (fDecRet == -100)
                break;
            else if (fDecRet == 0)
                stack[++stackTop] = faninId;
            assert(stackTop < STACK_SIZE);

            // printf("  iteration end\n");
        }

        if (vCutSizes[rootId] != -1) {
            // add all nodes in the trav list with ref > 0 into the cut list
            for (int i = 0; i < travSize; i++) {
                assert(travRefs[i] >= 0);
                if (travRefs[i] == 0) continue;

                if (vCutSizes[rootId] < nMaxCutSize) {
                    // add into cut list
                    vCutTable[rootId * CUT_TABLE_SIZE + vCutSizes[rootId]] = travIds[i];
                    vCutSizes[rootId]++;
                } else {
                    // the cut has reached max size
                    vCutSizes[rootId] = -1;
                    break;
                }
            }
            assert(vCutSizes[rootId] <= MAX_CUT_SIZE);
            // save coneSize
            vConeSizes[rootId] = coneSize;
        }

        // printf("node: %d, cone size: %d | ", rootId, vConeSizes[rootId]);
        // for (int j = 0; j < vCutSizes[rootId]; j++) {
        //     printf("%d ", vCutTable[rootId * CUT_TABLE_SIZE + j]);
        // }
        // printf("\n");
    }
}

__device__ int decrementReconvRef(int rootId, int nodeId, const int * pNumFanouts, 
                                  const int * vCutTable, const int * vCutSizes, 
                                  int * pTravSize, int * travIds, int * travRefs) {
    // terminate when reaching the cut nodes
    for (int i = 0; i < vCutSizes[rootId]; i++)
        if (vCutTable[rootId * CUT_TABLE_SIZE + i] == nodeId)
            return 1;
    
    // check whether nodeId is already in the trav list
    for (int i = 0; i < *pTravSize; i++)
        if (travIds[i] == nodeId)
            return --travRefs[i];
    assert(*pTravSize < STACK_SIZE);

    // nodeId is not in the trav list; insert it
    travIds[*pTravSize] = nodeId;
    travRefs[*pTravSize] = pNumFanouts[nodeId] - 1;
    (*pTravSize)++;
    return pNumFanouts[nodeId] - 1;
}

__global__ void getReconvNumSaved(const int * pFanin0, const int * pFanin1, const int * pNumFanouts, 
                                  const int * vReconvInd, const int * vCutTable, const int * vCutSizes, 
                                  int * vNumSaved, int nReconv) {
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stack[STACK_SIZE], travIds[STACK_SIZE], travRefs[STACK_SIZE];
    int stackTop, travSize, savedSize;
    int nodeId, rootId, faninId;

    for (; idx < nReconv; idx += nThreads) {
        rootId = vReconvInd[idx];
        stackTop = -1, travSize = 0, savedSize = 0;
        stack[++stackTop] = rootId;

        while (stackTop != -1) {
            nodeId = stack[stackTop--];
            savedSize++;

            // check two fanins
            faninId = dUtils::AigNodeID(pFanin1[nodeId]);
            if (decrementReconvRef(rootId, faninId, pNumFanouts, vCutTable, vCutSizes, 
                                   &travSize, travIds, travRefs) == 0) {
                assert(stackTop < STACK_SIZE);
                stack[++stackTop] = faninId;
            }

            faninId = dUtils::AigNodeID(pFanin0[nodeId]);
            if (decrementReconvRef(rootId, faninId, pNumFanouts, vCutTable, vCutSizes, 
                                   &travSize, travIds, travRefs) == 0) {
                assert(stackTop < STACK_SIZE);
                stack[++stackTop] = faninId;
            }
        }
        vNumSaved[rootId] = savedSize;
    }
}

__device__ int getReconvCutIter(int rootId,
                                const int * pFanin0, const int * pFanin1, 
                                const int * pNumFanouts, const int * pLevels, 
                                int * visited, int * pVisitedSize,
                                int * vCutTable, int * vCutSizes, 
                                int nPIs, int nMaxCutSize, int nFanoutLimit) {
    int nodeId, faninId, bestId = -1, bestIdx = -1;
    int bestCost = 100, currCost;
    int fFanin0Visited, fFanin1Visited, fBestFanin0Visited = 0, fBestFanin1Visited = 0;

    // find the best cost cut node to expand
    for (int i = 0; i < vCutSizes[rootId]; i++) {
        nodeId = vCutTable[rootId * CUT_TABLE_SIZE + i];

        // get the number of new leaves
        fFanin0Visited = fFanin1Visited = 0;
        if (dUtils::AigIsPIConst(nodeId, nPIs))
            currCost = 999;
        else {
            faninId = dUtils::AigNodeID(pFanin0[nodeId]);
            for (int j = 0; j < *pVisitedSize; j++)
                if (visited[j] == faninId) {
                    fFanin0Visited = 1;
                    break;
                }
            
            faninId = dUtils::AigNodeID(pFanin1[nodeId]);
            for (int j = 0; j < *pVisitedSize; j++)
                if (visited[j] == faninId) {
                    fFanin1Visited = 1;
                    break;
                }
            
            currCost = (1 - fFanin0Visited) + (1 - fFanin1Visited);
            if (currCost >= 2) {
                if (pNumFanouts[nodeId] > nFanoutLimit)
                    currCost = 999;
            }
        }

        // update best node
        if (bestCost > currCost || (bestCost == currCost && pLevels[nodeId] > pLevels[bestId])) {
            bestCost = currCost, bestId = nodeId, bestIdx = i;
            fBestFanin0Visited = fFanin0Visited, fBestFanin1Visited = fFanin1Visited;
        }
        if (bestCost == 0)
            break;
    }

    if (bestId == -1)
        return 0;
    assert(bestCost < 3);

    if (vCutSizes[rootId] - 1 + bestCost > nMaxCutSize)
        return 0;
    assert(dUtils::AigIsNode(bestId, nPIs));

    // remove the best node from cut list
    for (int i = bestIdx + 1; i < vCutSizes[rootId]; i++)
        vCutTable[rootId * CUT_TABLE_SIZE + i - 1] = vCutTable[rootId * CUT_TABLE_SIZE + i];
    vCutSizes[rootId]--;

    if (!fBestFanin0Visited) {
        assert(*pVisitedSize < STACK_SIZE);
        faninId = dUtils::AigNodeID(pFanin0[bestId]);
        vCutTable[rootId * CUT_TABLE_SIZE + (vCutSizes[rootId]++)] = faninId;
        visited[(*pVisitedSize)++] = faninId;
    }
    if (!fBestFanin1Visited) {
        assert(*pVisitedSize < STACK_SIZE);
        faninId = dUtils::AigNodeID(pFanin1[bestId]);
        vCutTable[rootId * CUT_TABLE_SIZE + (vCutSizes[rootId]++)] = faninId;
        visited[(*pVisitedSize)++] = faninId;
    }
    assert(vCutSizes[rootId] <= nMaxCutSize);
    return 1;
}

__global__ void getReconvCut(const int * pFanin0, const int * pFanin1, 
                             const int * pNumFanouts, const int * pLevels, 
                             const int * vReconvInd, int * vCutTable, int * vCutSizes, 
                             int nReconv, int nPIs, int nMaxCutSize, int nFanoutLimit) {
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int visited[STACK_SIZE];
    int visitedSize;
    int rootId, faninId;

    for (; idx < nReconv; idx += nThreads) {
        visitedSize = 0;
        rootId = vReconvInd[idx];
        vCutSizes[rootId] = 0;

        // initialize the cut list and visited list
        visited[visitedSize++] = rootId;

        faninId = dUtils::AigNodeID(pFanin0[rootId]);
        vCutTable[rootId * CUT_TABLE_SIZE + (vCutSizes[rootId]++)] = faninId;
        visited[visitedSize++] = faninId;

        faninId = dUtils::AigNodeID(pFanin1[rootId]);
        vCutTable[rootId * CUT_TABLE_SIZE + (vCutSizes[rootId]++)] = faninId;
        visited[visitedSize++] = faninId;

        // iteratively expand the cut
        while (getReconvCutIter(rootId, pFanin0, pFanin1, pNumFanouts, pLevels, visited, &visitedSize,
                   vCutTable, vCutSizes, nPIs, nMaxCutSize, nFanoutLimit));
        assert(vCutSizes[rootId] <= nMaxCutSize);
    }
}

__global__ void getTruthWordNums(const int * vResynInd, const int * vCutSizes, 
                                 int * vTruthRanges, int nResyn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nResyn) {
        int nodeId = vResynInd[idx];
        assert(vCutSizes[nodeId] > 0);
        vTruthRanges[idx] = dUtils::TruthWordNum(vCutSizes[nodeId]);
    }
}


__global__ void insertChoiceGraphs(const uint64 * vSubgTable, const int * vSubgLinks, const int * vSubgLens, 
                                   const int * vResynInd, const int * vCutTable, const int * vCutSizes, 
                                   uint64 * htKeys, uint32 * htValues, int htCapacity, int * vChoicesLit, 
                                   int nResyn, int nObjs, int nPIs, int * pnNewObjs) {
    // vChoicesLit should be initialized as -1
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId, subgId, newId, fanin0Id, fanin1Id;
    int nCutSize, nSubgSize;
    int currRowIdx, columnPtr;
    int mapId[MAX_SUBG_SIZE]; // map subgraph id of a node to old node id in the original graph
    uint32 nodeId, realNewId;
    int lit0, lit1, lit2, lit3, fComp, temp;
    uint64 key;

    for (; idx < nResyn; idx += nThreads) {
        rootId = vResynInd[idx];
        nSubgSize = vSubgLens[idx];
        nCutSize = vCutSizes[rootId];

        assert(nSubgSize + nCutSize <= MAX_SUBG_SIZE);
        // nSubgSize == 0 means that in pre-evaluation the resyned subgraphs are no better than the original one
        // vChoicesLit is still -1
        if (nSubgSize == 0)
            continue;

        for (int i = nCutSize; i < nCutSize + nSubgSize; i++)
            mapId[i] = -1;
        for (int i = 0; i < nCutSize; i++)
            mapId[i] = vCutTable[rootId * CUT_TABLE_SIZE + i];
        
        // printf(" *** root id %d, cut nodes: ", rootId);
        // for (int k = 0; k < nCutSize; k++)
        //     printf("%d ", mapId[k]);
        // printf("| subg: ");
        // currRowIdx = idx, columnPtr = 0;
        // for (int k = nCutSize; k < nCutSize + nSubgSize; k++) {
        //     if (columnPtr == SUBG_TABLE_SIZE) {
        //         columnPtr = 0;
        //         currRowIdx = vSubgLinks[currRowIdx];
        //         assert(currRowIdx > 0);
        //     }
        //     subgUtil::unbindAndNodeKeyFlag(vSubgTable[currRowIdx * SUBG_TABLE_SIZE + (columnPtr++)], 
        //                             &lit0, &lit1, &fComp);
        //     fanin0Id = dUtils::AigNodeID(lit0);
        //     fanin1Id = dUtils::AigNodeID(lit1);
        //     printf("%s%d,%s%d ", dUtils::AigNodeIsComplement(lit0) ? "!" : "", fanin0Id, dUtils::AigNodeIsComplement(lit1) ? "!" : "", fanin1Id);
        //     if (k == nCutSize + nSubgSize - 1)
        //         printf(" root complemented: %s", fComp ? "y" : "n");
        // }
        // printf("\n");

        // take care of the case that the resyned cut is a const or a single var of cut nodes
        if (nSubgSize == 1) {
            subgUtil::unbindAndNodeKeyFlag(vSubgTable[idx * SUBG_TABLE_SIZE], &lit0, &lit1, &fComp);
            // having two same lits is the indicator of these cases
            if (lit0 == lit1) {
                fanin0Id = dUtils::AigNodeID(lit0); // in this case lit0 is using the original id
                assert(fanin0Id < nObjs);

                // debug assertion
                // int fCutNode = 0;
                // for (int i = 0; i < nCutSize; i++)
                //     if (mapId[i] == fanin0Id) {
                //         fCutNode = 1;
                //         break;
                //     }
                // assert(fCutNode || fanin0Id == 0);

                assert(fComp == dUtils::AigNodeIsComplement(lit0));
                vChoicesLit[idx] = lit0;
                continue;
            }
        }

        // in topo order of the subgraph, check whether each node has corresponding old node
        currRowIdx = idx, columnPtr = 0;
        for (int i = 0; i < nSubgSize; i++) {
            // change row
            if (columnPtr == SUBG_TABLE_SIZE) {
                columnPtr = 0;
                currRowIdx = vSubgLinks[currRowIdx];
            }

            subgId = i + nCutSize; // new id of the node in subgraph
            subgUtil::unbindAndNodeKeyFlag(vSubgTable[currRowIdx * SUBG_TABLE_SIZE + (columnPtr++)], 
                                           &lit0, &lit1, &fComp);
            fanin0Id = dUtils::AigNodeID(lit0);
            fanin1Id = dUtils::AigNodeID(lit1);
            // if one of the fanins is a new node, then this is also a new node
            if (mapId[fanin0Id] == -1 || mapId[fanin1Id] == -1)
                continue;
            
            // note that the old node will never be a PI, since this is an AND node in the new subgraph
            assert(lit0 <= lit1);
            lit2 = dUtils::AigNodeLitCond(mapId[fanin0Id], dUtils::AigNodeIsComplement(lit0));
            lit3 = dUtils::AigNodeLitCond(mapId[fanin1Id], dUtils::AigNodeIsComplement(lit1));
            if (lit2 > lit3)
                temp = lit2, lit2 = lit3, lit3 = temp;
            key = formAndNodeKey(lit2, lit3);
            nodeId = retrieve_single_volatile<uint64, uint32>(htKeys, htValues, key, htCapacity);
            // nodeId = retrieve_single<uint64, uint32>(htKeys, htValues, key, htCapacity);
            if (nodeId != (HASHTABLE_EMPTY_VALUE<uint64, uint32>) && (int)nodeId > 0 && (int)nodeId < nObjs)
                mapId[subgId] = (int)nodeId;
            // nodeId < nObjs is to filter out the subsequently inserted new nodes by other threads
        }

        if (mapId[nCutSize + nSubgSize - 1] != -1) {
            // the whole new subgraph is the same as the old one
            // note, vChoicesLit is still -1 for this root
            // if (mapId[nCutSize + nSubgSize - 1] != rootId && rootId == 203) {
            //     printf(" *** error: root id %d, subg ", rootId);
            //     currRowIdx = idx, columnPtr = 0;
            //     for (int k = nCutSize; k < nCutSize + nSubgSize; k++) {
            //         if (columnPtr == SUBG_TABLE_SIZE) {
            //             columnPtr = 0;
            //             currRowIdx = vSubgLinks[currRowIdx];
            //             assert(currRowIdx > 0);
            //         }
            //         subgUtil::unbindAndNodeKeyFlag(vSubgTable[currRowIdx * SUBG_TABLE_SIZE + (columnPtr++)], 
            //                                &lit0, &lit1, &fComp);
            //         fanin0Id = dUtils::AigNodeID(lit0);
            //         fanin1Id = dUtils::AigNodeID(lit1);
            //         printf("%s%d,%s%d ", dUtils::AigNodeIsComplement(lit0) ? "!" : "", fanin0Id, dUtils::AigNodeIsComplement(lit1) ? "!" : "", fanin1Id);
            //     }
            //     printf("; mapId: ");
            //     for (int k = 0; k < nCutSize; k++)
            //         printf("%d ", mapId[k]);
            //     printf("| ");
            //     for (int k = nCutSize; k < nCutSize + nSubgSize; k++)
            //         printf("%d ", mapId[k]);
            //     printf("\n");
            //     assert(0);
            // }
            // assert(mapId[nCutSize + nSubgSize - 1] == rootId);
            // printf("   found whole new subgraph has nodes in the original graph, original root = %d, chioce root = %d\n", rootId, mapId[nCutSize + nSubgSize - 1]);
            
            // FIXME if its not the same as root, we found a real pair of choice nodes between two old nodes
            if (mapId[nCutSize + nSubgSize - 1] != rootId) {
                assert(mapId[nCutSize + nSubgSize - 1] < nObjs);
                vChoicesLit[idx] = dUtils::AigNodeLitCond(mapId[nCutSize + nSubgSize - 1], fComp);
            }
            continue;
        }

        __threadfence();

        // printf("** begin insertion of root id %d, subg ", rootId);
        // currRowIdx = idx, columnPtr = 0;
        // for (int k = nCutSize; k < nCutSize + nSubgSize; k++) {
        //     if (columnPtr == SUBG_TABLE_SIZE) {
        //         columnPtr = 0;
        //         currRowIdx = vSubgLinks[currRowIdx];
        //         assert(currRowIdx > 0);
        //     }
        //     subgUtil::unbindAndNodeKeyFlag(vSubgTable[currRowIdx * SUBG_TABLE_SIZE + (columnPtr++)], 
        //                             &lit0, &lit1, &fComp);
        //     fanin0Id = dUtils::AigNodeID(lit0);
        //     fanin1Id = dUtils::AigNodeID(lit1);
        //     printf("%s%d,%s%d ", dUtils::AigNodeIsComplement(lit0) ? "!" : "", fanin0Id, dUtils::AigNodeIsComplement(lit1) ? "!" : "", fanin1Id);
        // }
        // printf("; mapId: ");
        // for (int k = 0; k < nCutSize; k++)
        //     printf("%d ", mapId[k]);
        // printf("| ");
        // for (int k = nCutSize; k < nCutSize + nSubgSize; k++)
        //     printf("%d ", mapId[k]);
        // printf("\n");

        // add new nodes without corresponding old nodes to the hash table
        currRowIdx = idx, columnPtr = 0;
        for (int i = 0; i < nSubgSize; i++) {
            // change row
            if (columnPtr == SUBG_TABLE_SIZE) {
                columnPtr = 0;
                currRowIdx = vSubgLinks[currRowIdx];
                assert(currRowIdx > 0);
            }

            subgUtil::unbindAndNodeKeyFlag(vSubgTable[currRowIdx * SUBG_TABLE_SIZE + (columnPtr++)], 
                                           &lit0, &lit1, &fComp);
            fanin0Id = dUtils::AigNodeID(lit0);
            fanin1Id = dUtils::AigNodeID(lit1);

            subgId = i + nCutSize;
            if (mapId[subgId] != -1)
                continue;
            
            // if (mapId[fanin0Id] == -1 || mapId[fanin1Id] == -1) {
            //     printf(" *** error (curr i=%d): root id %d, subg ", i, rootId);
            //     currRowIdx = idx, columnPtr = 0;
            //     for (int k = nCutSize; k < nCutSize + nSubgSize; k++) {
            //         if (columnPtr == SUBG_TABLE_SIZE) {
            //             columnPtr = 0;
            //             currRowIdx = vSubgLinks[currRowIdx];
            //             assert(currRowIdx > 0);
            //         }
            //         subgUtil::unbindAndNodeKeyFlag(vSubgTable[currRowIdx * SUBG_TABLE_SIZE + (columnPtr++)], 
            //                                &lit0, &lit1, &fComp);
            //         fanin0Id = dUtils::AigNodeID(lit0);
            //         fanin1Id = dUtils::AigNodeID(lit1);
            //         printf("%s%d,%s%d ", dUtils::AigNodeIsComplement(lit0) ? "!" : "", fanin0Id, dUtils::AigNodeIsComplement(lit1) ? "!" : "", fanin1Id);
            //     }
            //     printf("; mapId: ");
            //     for (int k = 0; k < nCutSize; k++)
            //         printf("%d ", mapId[k]);
            //     printf("| ");
            //     for (int k = nCutSize; k < nCutSize + nSubgSize; k++)
            //         printf("%d ", mapId[k]);
            //     printf("\n");
            //     assert(0);
            // }

            assert(mapId[fanin0Id] != -1 && mapId[fanin1Id] != -1);

            newId = atomicAdd(pnNewObjs, 1);
            lit2 = dUtils::AigNodeLitCond(mapId[fanin0Id], dUtils::AigNodeIsComplement(lit0));
            lit3 = dUtils::AigNodeLitCond(mapId[fanin1Id], dUtils::AigNodeIsComplement(lit1));
            if (lit2 > lit3)
                temp = lit2, lit2 = lit3, lit3 = temp;
            key = formAndNodeKey(lit2, lit3);
            uint32 hashRet = insert_single_no_update_volatile<uint64, uint32>(htKeys, htValues, key, (uint32)newId, htCapacity);
            // uint32 hashRet = insert_single_no_update<uint64, uint32>(htKeys, htValues, key, (uint32)newId, htCapacity);

            __threadfence(); // ensure strong memory order
            
            // immediate retrieval to check whether a same node is also created by other threads
            // NOTE the assignments of key and value are not in a single atomic transaction,
            // so it could happen that when another thread has the same key, 
            // the key is already assigned in hashtable but the value is not, so the retrieved value is still empty_value
            do {
                realNewId = retrieve_single_volatile<uint64, uint32>(htKeys, htValues, key, htCapacity);
                // realNewId = retrieve_single<uint64, uint32>(htKeys, htValues, key, htCapacity);
                __threadfence();
            } while (realNewId == (HASHTABLE_EMPTY_VALUE<uint64, uint32>));
            // realNewId = retrieve_single<uint64, uint32>(htKeys, htValues, key, htCapacity);
            // if ((int)realNewId < nObjs) {
            //     printf(" *** error (hash key=%llu, hash ret=%u, curr i=%d): root id %d, new id %u, real new id %u, subg ", key, hashRet, i, rootId, (uint32)newId, realNewId);
            //     currRowIdx = idx, columnPtr = 0;
            //     for (int k = nCutSize; k < nCutSize + nSubgSize; k++) {
            //         if (columnPtr == SUBG_TABLE_SIZE) {
            //             columnPtr = 0;
            //             currRowIdx = vSubgLinks[currRowIdx];
            //             assert(currRowIdx > 0);
            //         }
            //         subgUtil::unbindAndNodeKeyFlag(vSubgTable[currRowIdx * SUBG_TABLE_SIZE + (columnPtr++)], 
            //                                &lit0, &lit1, &fComp);
            //         fanin0Id = dUtils::AigNodeID(lit0);
            //         fanin1Id = dUtils::AigNodeID(lit1);
            //         printf("%s%d,%s%d ", dUtils::AigNodeIsComplement(lit0) ? "!" : "", fanin0Id, dUtils::AigNodeIsComplement(lit1) ? "!" : "", fanin1Id);
            //     }
            //     printf("; mapId: ");
            //     for (int k = 0; k < nCutSize; k++)
            //         printf("%d ", mapId[k]);
            //     printf("| ");
            //     for (int k = nCutSize; k < nCutSize + nSubgSize; k++)
            //         printf("%d ", mapId[k]);
            //     printf("\n");
            //     assert(0);
            // }
            
            assert((int)realNewId >= nObjs);
            assert(realNewId != (HASHTABLE_EMPTY_VALUE<uint64, uint32>));

            // update id mapping since this new node is now in the hash table
            mapId[subgId] = (int)realNewId;
            
            // printf("  Attempted newId %u, realNewId %u %s; mapId: ", newId, realNewId, newId == realNewId ? "(added)" : "       ");
            // for (int k = 0; k < nCutSize; k++)
            //     printf("%d ", mapId[k]);
            // printf("| ");
            // for (int k = nCutSize; k <= nCutSize + i; k++)
            //     printf("%d ", mapId[k]);
            // printf("\n");
        }
        // save choice node lit of the root
        assert(mapId[nCutSize + nSubgSize - 1] == (int)realNewId);
        vChoicesLit[idx] = dUtils::AigNodeLitCond((int)realNewId, fComp);
        // printf("  Choice Lit: %d\n", vChoicesLit[idx]);
    }
}

__global__ void unbindAllKeys(const uint64 * vReconstructedKeys, const uint32 * vReconstructedIds,
                              int * vFanin0New, int * vFanin1New, int nEntries, int nObjs, int nBufferLen) {
    // NOTE the four pointers should point to the end of old nObjs, i.e., the begin of new objs
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // to ensure that the pointer is correctly pointed to the begin of new objs
    if (idx == 0) {
        assert(*vReconstructedIds >= nObjs);
        assert(*(vReconstructedIds - 1) < nObjs);
    }
    if (idx < nEntries) {
        uint64 key = vReconstructedKeys[idx];
        uint32 lit0, lit1, nodeId;
        unbindAndNodeKeys(key, &lit0, &lit1);
        nodeId = vReconstructedIds[idx];
        assert(nodeId < nBufferLen);

        // for a value that does not correspond to a valid nodeId, its fanin = -1
        vFanin0New[nodeId - nObjs] = (int)lit0;
        vFanin1New[nodeId - nObjs] = (int)lit1;
    }
}

inline int isRedundantNode(int nodeId, int nPIs, const int * fanin0, const int * fanin1) {
    // created during choice, where the fanin0 is const true, and this node is the same as its fanin1
    // ** fanin0[nodeId] == fanin1[nodeId] 
    return nodeId > nPIs && fanin0[nodeId] == dUtils::AigConst1;
}

int topoSort(int nodeId, int nPIs, int * levels, const int * fanin0, const int * fanin1) {
    // printf("  Topo sorting nodeId %d ...\n", nodeId);
    if(levels[nodeId] != -1)
        return levels[nodeId];
    if (isRedundantNode(nodeId, nPIs, fanin0, fanin1))
        return (levels[nodeId] = 
                topoSort(AigNodeID(fanin1[nodeId]), nPIs, levels, fanin0, fanin1));
    return (levels[nodeId] = 
            1 + max(
                topoSort(AigNodeID(fanin0[nodeId]), nPIs, levels, fanin0, fanin1),
                topoSort(AigNodeID(fanin1[nodeId]), nPIs, levels, fanin0, fanin1)
            ));
}

std::tuple<int, int *, int *, int *, int>
choiceReorder(int * vFanin0New, int * vFanin1New, int * vResynInd, int * vChoicesLit,
              int * vCutTable, int * vCutSizes, int * pNumFanouts, int * pOuts, 
              int nPIs, int nPOs, int nObjs, int nBufferLen, int nResyn, bool fUseZeros) {
    int nNodesNew, nObjsNew;
    int temp;

    // copy fanin, vChoicesLit data to host
    int * vhFanin0, * vhFanin1, * vhResynInd, * vhChoicesLit, * vhCutTable, * vhCutSizes;
    int * vhDeleted, * vhNumFanouts;
    int * vhLevels, * vhNewInd;
    int * vhFanin0New, * vhFanin1New, * vhOutsNew;
    int * vhOldEquivChoices; // record the case that two old nodes are choices of each other, to prevent recurrent reference
    // vhFanin0 = (int *) malloc(nBufferLen * sizeof(int));
    // vhFanin1 = (int *) malloc(nBufferLen * sizeof(int));
    // vhResynInd = (int *) malloc(nResyn * sizeof(int));
    // vhChoicesLit = (int *) malloc(nResyn * sizeof(int));
    // vhCutTable = (int *) malloc(nObjs * CUT_TABLE_SIZE * sizeof(int));
    // vhCutSizes = (int *) malloc(nObjs * sizeof(int));
    // vhDeleted = (int *) malloc(nBufferLen * sizeof(int));
    // vhNumFanouts = (int *) malloc(nBufferLen * sizeof(int));
    // vhLevels = (int *) malloc(nBufferLen * sizeof(int));
    // vhNewInd = (int *) malloc(nBufferLen * sizeof(int));
    cudaHostAlloc(&vhFanin0, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhFanin1, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhResynInd, nResyn * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhChoicesLit, nResyn * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhCutTable, nObjs * CUT_TABLE_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhCutSizes, nObjs * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhDeleted, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhNumFanouts, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhLevels, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhNewInd, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaMemcpy(vhFanin0, vFanin0New, nBufferLen * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vhFanin1, vFanin1New, nBufferLen * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vhResynInd, vResynInd, nResyn * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vhChoicesLit, vChoicesLit, nResyn * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vhCutTable, vCutTable, nObjs * CUT_TABLE_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vhCutSizes, vCutSizes, nObjs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    memset(vhDeleted, 0, nObjs * sizeof(int));
    memcpy(vhNumFanouts, pNumFanouts, nObjs * sizeof(int));

    vhOldEquivChoices = (int *) malloc(nObjs * sizeof(int));
    memset(vhOldEquivChoices, -1, nObjs * sizeof(int));

    printf("Allocated and copied data\n");
    // int maxChoiceLit = thrust::reduce(thrust::device, vChoicesLit, vChoicesLit + nResyn, 0, thrust::maximum<int>());
    // printf("maximum choice id: %d (nBufferLen=%d)\n", AigNodeID(maxChoiceLit), nBufferLen);

    // debug assertion
    // for (int i = nPIs + 1; i < nBufferLen; i++) {
    //     // if (i == nObjs)
    //     //     printf("----------\n");
    //     // if (vhFanin0[i] == -1) {
    //     //     printf("%d\n", i);
    //     //     continue;
    //     // }
    //     // printf("%d\t", i);
    //     // printf("%s%d\t", (vhFanin0[i] & 1) ? "!" : "", vhFanin0[i] >> 1);
    //     // printf("%s%d\n", (vhFanin1[i] & 1) ? "!" : "", vhFanin1[i] >> 1);

    //     if (vhFanin0[i] != -1) {
    //         assert(vhFanin1[i] != -1);
    //         if (AigNodeID(vhFanin0[i]) > nPIs)
    //             assert(vhFanin0[AigNodeID(vhFanin0[i])] != -1);
    //         if (AigNodeID(vhFanin1[i]) > nPIs)
    //             assert(vhFanin0[AigNodeID(vhFanin1[i])] != -1);
    //     }
    // }
    // for (int i = 0; i < nResyn; i++) {
    //     int lit = vhChoicesLit[i];
    //     if (lit != -1 && AigNodeID(lit) > nPIs) {
    //         assert(vhFanin0[AigNodeID(lit)] != -1);
    //         assert(vhFanin1[AigNodeID(lit)] != -1);
    //     }
    // }

    auto cpuSequentialStartTime = hrClock();

    // initialize the new nodes as deleted
    for (int i = nObjs; i < nBufferLen; i++)
        vhDeleted[i] = 1, vhNumFanouts[i] = 0;
    
    std::function<int(int)> installNode = [&](int nodeId) {
        if (vhDeleted[nodeId]) {
            // printf("    install %d\n", nodeId);
            vhDeleted[nodeId] = 0;
            int cnt = !isRedundantNode(nodeId, nPIs, vhFanin0, vhFanin1);
            int fanin0Id = AigNodeID(vhFanin0[nodeId]);
            int fanin1Id = AigNodeID(vhFanin1[nodeId]);
            cnt += installNode(fanin0Id);
            cnt += installNode(fanin1Id);
            vhNumFanouts[fanin0Id]++;
            vhNumFanouts[fanin1Id]++;
            return cnt;
        } else
            return 0;
    };

    std::function<int(int)> deleteNode = [&](int nodeId) {
        if (nodeId <= nPIs) {
            // this also happens when recursive calling from a redundant node
            // assert(nodeId == AigNodeID(dUtils::AigConst1));
            return 0;
        }

        // printf("    delete %d, ", nodeId);
        assert(vhDeleted[nodeId] == 0);
        int fanin0Id = AigNodeID(vhFanin0[nodeId]);
        int fanin1Id = AigNodeID(vhFanin1[nodeId]);
        vhNumFanouts[fanin0Id]--;
        vhNumFanouts[fanin1Id]--;
        // printf("fanin0 %d #fanout=%d, fanin1 %d #fanout=%d\n", fanin0Id, vhNumFanouts[fanin0Id], fanin1Id, vhNumFanouts[fanin1Id]);
        int cnt = !isRedundantNode(nodeId, nPIs, vhFanin0, vhFanin1);
        if (vhNumFanouts[fanin0Id] == 0)
            cnt += deleteNode(fanin0Id);
        if (vhNumFanouts[fanin1Id] == 0)
            cnt += deleteNode(fanin1Id);
        vhDeleted[nodeId] = 1;
        return cnt;
    };

    int nReplaced = 0;
    // for (int i = 0; i < nResyn; i++) {
    for (int i = nResyn - 1; i >= 0; i--) {
        int nodeId = vhResynInd[i];
        int choiceLit = vhChoicesLit[i];
        if (vhDeleted[nodeId]) continue;
        // the resyned cone is identical to the original structure, 
        // or in pre-eval the resyned cone is worse than the original one
        if (choiceLit == -1) continue;

        int choiceId = AigNodeID(choiceLit);
        // ** if (newIdx == i || id(fanin1[newIdx]) == i) continue;
        int cutIntegrity = 1;
        for (int j = 0; j < vhCutSizes[nodeId]; j++)
            if (vhDeleted[vhCutTable[nodeId * CUT_TABLE_SIZE + j]]) {
                cutIntegrity = 0;
                break;
            }
        if (!cutIntegrity) continue;
        // printf("resyn node %d, choice %d\n", nodeId, choiceId);
        // printf("  fanin of node %d: %s%d\t%s%d\n", nodeId, (vhFanin0[nodeId] & 1) ? "!" : "", vhFanin0[nodeId] >> 1, (vhFanin1[nodeId] & 1) ? "!" : "", vhFanin1[nodeId] >> 1);
        // printf("  fanin of choice %d: %s%d\t%s%d\n", choiceId, (vhFanin0[choiceId] & 1) ? "!" : "", vhFanin0[choiceId] >> 1, (vhFanin1[choiceId] & 1) ? "!" : "", vhFanin1[choiceId] >> 1);

        int nAdded, nSaved;
        if (choiceId < nObjs) {
            // the case that the resyned cut (choice node) is a const or a single var of cut nodes,
            // or every node in the resyned cone has correspondence in the original structure (incl. root)
            // no new nodes will be added
            // printf("Warning: encountered the case that the resyned cut is const 0/1 or a single var of cut nodes\n");
            if (vhDeleted[choiceId])
                continue;
            if (vhOldEquivChoices[choiceId] == nodeId) {
                // assert(choiceId < nodeId);
                printf("Detected a pair of recurrent old choice nodes (%d, %d), skip %d\n", choiceId, nodeId, nodeId);
                continue;
            }
            nAdded = 0;
        } else {
            nAdded = installNode(choiceId);
            // printf("  finished install choice\n");
        }
        vhNumFanouts[choiceId]++;
        nSaved = deleteNode(nodeId);
        vhNumFanouts[choiceId]--;
        assert(!vhDeleted[choiceId]);
        // printf("  finished delete node\n");

        if (nSaved - nAdded > 0 || (nSaved == nAdded && fUseZeros)) {
            // the choice cone is better
            // make the original root as a redundant node of the choice root
            // note, all of the fanouts of nodeId remain valid, since the function of nodeId is not changed
            vhDeleted[nodeId] = 0;
            vhFanin0[nodeId] = dUtils::AigConst1;
            vhFanin1[nodeId] = choiceLit;
            vhNumFanouts[AigNodeID(dUtils::AigConst1)]++;
            vhNumFanouts[choiceId]++;
            nReplaced++;
            // printf("  choice is better\n");
            // for the no new node case, this will be a redundant node of const 0/1 or a cut node
            if (choiceId < nObjs) {
                vhOldEquivChoices[nodeId] = choiceId;
            }
        } else {
            // the original cone is better, revert
            installNode(nodeId);

            assert(vhNumFanouts[choiceId] == 0);
            vhNumFanouts[nodeId]++;
            deleteNode(choiceId);
            vhNumFanouts[nodeId]--;
            // printf("  original is better, reverted\n");
        }
        // note, no matter which cone is installed, for all node in vhResynInd, vhDeleted is always false
        assert(!vhDeleted[nodeId]);
        // printf("  finished this resyn iter\n");
    }
    printf("Total number of replaced cones: %d, resyned cones: %d\n", nReplaced, nResyn);
    // return {NULL, NULL, NULL, NULL};

    // remove redundancy
    // topo order to get level of each node. the redundant node does not contribute to level
    memset(vhLevels, -1, nBufferLen * sizeof(int));
    for (int i = 0; i <= nPIs; i++)
        vhLevels[i] = 0;
    for (int i = nPIs + 1; i < nBufferLen; i++)
        if (!vhDeleted[i]) {
            // printf("Topo sorting nodeId %d ...\n", i);
            topoSort(i, nPIs, vhLevels, vhFanin0, vhFanin1);
        }

    // count total number of nodes and assign each node an id level by level
    int nMaxLevel = 0;
    std::vector<int> vLevelNodesCount(1, 0);
    for (int i = nPIs + 1; i < nBufferLen; i++)
        if (!vhDeleted[i] && !isRedundantNode(i, nPIs, vhFanin0, vhFanin1)) {
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
    
    for (int i = nBufferLen - 1; i > nPIs; i--)
        if (!vhDeleted[i] && !isRedundantNode(i, nPIs, vhFanin0, vhFanin1))
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
        if (!vhDeleted[i] && !isRedundantNode(i, nPIs, vhFanin0, vhFanin1)) {
            assert(vhFanin0New[vhNewInd[i]] == -1 && vhFanin1New[vhNewInd[i]] == -1);
            // propagate if fanin is redundant
            int lit, propLit;
            if (isRedundantNode(AigNodeID(vhFanin0[i]), nPIs, vhFanin0, vhFanin1)) {
                propLit = vhFanin1[AigNodeID(vhFanin0[i])]; // lit of the redundant node
                propLit = dUtils::AigNodeLitCond(vhNewInd[AigNodeID(propLit)], AigNodeIsComplement(propLit));
                lit = dUtils::AigNodeNotCond(propLit, AigNodeIsComplement(vhFanin0[i]));
            } else
                lit = dUtils::AigNodeLitCond(vhNewInd[AigNodeID(vhFanin0[i])], 
                                             AigNodeIsComplement(vhFanin0[i]));
            vhFanin0New[vhNewInd[i]] = lit;

            if (isRedundantNode(AigNodeID(vhFanin1[i]), nPIs, vhFanin0, vhFanin1)) {
                propLit = vhFanin1[AigNodeID(vhFanin1[i])]; // lit of the redundant node
                propLit = dUtils::AigNodeLitCond(vhNewInd[AigNodeID(propLit)], AigNodeIsComplement(propLit));
                lit = dUtils::AigNodeNotCond(propLit, AigNodeIsComplement(vhFanin1[i]));
            } else
                lit = dUtils::AigNodeLitCond(vhNewInd[AigNodeID(vhFanin1[i])], 
                                             AigNodeIsComplement(vhFanin1[i]));
            vhFanin1New[vhNewInd[i]] = lit;

            if (vhFanin0New[vhNewInd[i]] > vhFanin1New[vhNewInd[i]]) {
                temp = vhFanin0New[vhNewInd[i]];
                vhFanin0New[vhNewInd[i]] = vhFanin1New[vhNewInd[i]];
                vhFanin1New[vhNewInd[i]] = temp;
            }
        }
    
    // for (int i = 0; i < nObjsNew; i++) {
    //     assert(AigNodeID(vhFanin0New[i]) < i);
    //     assert(AigNodeID(vhFanin1New[i]) < i);
    //     if (vhFanin0New[i] == -1) {
    //         assert(vhFanin1New[i] == -1);
    //         assert(i <= nPIs);
    //         printf("%d\n", i);
    //         continue;
    //     }
    //     printf("%d\t", i);
    //     printf("%s%d\t", (vhFanin0New[i] & 1) ? "!" : "", vhFanin0New[i] >> 1);
    //     printf("%s%d\n", (vhFanin1New[i] & 1) ? "!" : "", vhFanin1New[i] >> 1);
    // }

    // update POs
    for (int i = 0; i < nPOs; i++) {
        int oldId = AigNodeID(pOuts[i]);
        int lit, propLit;
        assert(!vhDeleted[oldId]);

        if (isRedundantNode(oldId, nPIs, vhFanin0, vhFanin1)) {
            // propagate to the new node
            propLit = vhFanin1[oldId];
            propLit = dUtils::AigNodeLitCond(vhNewInd[AigNodeID(propLit)], AigNodeIsComplement(propLit));
            lit = dUtils::AigNodeNotCond(propLit, AigNodeIsComplement(pOuts[i]));
        } else
            lit = dUtils::AigNodeLitCond(vhNewInd[oldId], AigNodeIsComplement(pOuts[i]));
        
        vhOutsNew[i] = lit;
    }
    printf("Reordered network new nObjs: %d, original nObjs: %d\n", nObjsNew, nObjs);
    printf(" ** CPU sequential time: %.2lf sec\n", (hrClock() - cpuSequentialStartTime) / (double) NS_PER_SEC);

    // free(vhFanin0);
    // free(vhFanin1);
    // free(vhResynInd);
    // free(vhChoicesLit);
    // free(vhCutTable);
    // free(vhCutSizes);
    // free(vhDeleted);
    // free(vhNumFanouts);
    // free(vhLevels);
    // free(vhNewInd);
    cudaFreeHost(vhFanin0);
    cudaFreeHost(vhFanin1);
    cudaFreeHost(vhResynInd);
    cudaFreeHost(vhChoicesLit);
    cudaFreeHost(vhCutTable);
    cudaFreeHost(vhCutSizes);
    cudaFreeHost(vhDeleted);
    cudaFreeHost(vhNumFanouts);
    cudaFreeHost(vhLevels);
    cudaFreeHost(vhNewInd);
    free(vhOldEquivChoices);

    return {nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nMaxLevel};
}

std::tuple<int, int *, int *, int *, int> 
refactorPerform(bool fUseZeros, int cutSize,
                int nObjs, int nPIs, int nPOs, int nNodes, 
                const int * d_pFanin0, const int * d_pFanin1, const int * d_pOuts, 
                const int * d_pNumFanouts, const int * d_pLevel, 
                int * pOuts, int * pNumFanouts) {
    auto startTime = hrClock();

    int * vCutTable, * vCutSizes, * vNumSaved;
    int * vIdxSeq;
    int * vResynInd, * vReconvInd;
    int * vTruthRanges;
    unsigned * vTruth, * vTruthElem;
    uint64 * vSubgTable;
    int * vSubgLinks, * vSubgLens, * pSubgTableNext;
    int * vChoicesLit;
    int * vFanin0New, * vFanin1New;
    uint64 * vReconstructedKeys;
    uint32 * vReconstructedIds;

    cudaMalloc(&vCutTable, nObjs * CUT_TABLE_SIZE * sizeof(int));
    cudaMalloc(&vCutSizes, nObjs * sizeof(int));
    cudaMalloc(&vNumSaved, nObjs * sizeof(int));
    cudaMemset(vCutSizes, 0, nObjs * sizeof(int));

    cudaMalloc(&vIdxSeq, nObjs * sizeof(int));
    cudaMalloc(&vResynInd, nObjs * sizeof(int));

    // printAIG<<<1, 1>>>(d_pFanin0, d_pFanin1, d_pOuts, nNodes, nPIs, nPOs);
    // cudaDeviceSynchronize();

    // get mffc size
    getMffcCut<<<NUM_BLOCKS(nNodes, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        d_pFanin0, d_pFanin1, d_pNumFanouts, d_pLevel, 
        vCutTable, vCutSizes, vNumSaved, nNodes, nPIs, cutSize
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // auto printTimeStart = hrClock();
    // printMffc<<<1, 1>>>(vCutTable, vCutSizes, vNumSaved, d_pFanin0, d_pFanin1, nNodes, nPIs, nPOs);
    // cudaDeviceSynchronize();
    // auto printTime = hrClock() - printTimeStart;

    // filter out cones with too small size
    thrust::sequence(thrust::device, vIdxSeq, vIdxSeq + nObjs);
    int * pResynIndEnd = thrust::copy_if(
        thrust::device, 
        vIdxSeq + nPIs + 1, vIdxSeq + nObjs, 
        thrust::make_zip_iterator(thrust::make_tuple(vCutSizes + nPIs + 1, vNumSaved + nPIs + 1)),
        vResynInd,
        isNotSmallMffc()
    );
    cudaDeviceSynchronize();
    int nResyn = pResynIndEnd - vResynInd;
    printf("Num to resyn: %d\n", nResyn);
    if (nResyn == 0) {
        cudaFree(vCutTable);
        cudaFree(vCutSizes);
        cudaFree(vNumSaved);
        cudaFree(vIdxSeq);
        cudaFree(vResynInd);
        return {-1, NULL, NULL, NULL, -1};
    }

    // gather node with too large mffc cuts
    cudaMalloc(&vReconvInd, nResyn * sizeof(int));
    int * pReconvIndEnd = thrust::copy_if(
        thrust::device, 
        vIdxSeq + nPIs + 1, vIdxSeq + nObjs, 
        vCutSizes + nPIs + 1, 
        vReconvInd,
        dUtils::isMinusOne<int>()
    );
    cudaDeviceSynchronize();
    int nReconv = pReconvIndEnd - vReconvInd;
    printf("Num to compute reconv: %d\n", nReconv);
    assert(nReconv <= nResyn);

    // get reconvg cut
    if (nReconv > 0) {
        getReconvCut<<<NUM_BLOCKS(nReconv, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            d_pFanin0, d_pFanin1, d_pNumFanouts, d_pLevel, 
            vReconvInd, vCutTable, vCutSizes, nReconv, nPIs, cutSize, 50
        );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // compute number of nodes saved
        getReconvNumSaved<<<NUM_BLOCKS(nReconv, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            d_pFanin0, d_pFanin1, d_pNumFanouts, 
            vReconvInd, vCutTable, vCutSizes, vNumSaved, nReconv
        );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        // printTimeStart = hrClock();
        // printMffc<<<1, 1>>>(vCutTable, vCutSizes, vNumSaved, d_pFanin0, d_pFanin1, nNodes, nPIs, nPOs);
        // cudaDeviceSynchronize();
        // printTime += hrClock() - printTimeStart;
    }
    
    // allocate truth table for cuts to be resynthesized
    cudaMalloc(&vTruthRanges, nResyn * sizeof(int));
    getTruthWordNums<<<NUM_BLOCKS(nResyn, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vResynInd, vCutSizes, vTruthRanges, nResyn);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    thrust::inclusive_scan(thrust::device, vTruthRanges, vTruthRanges + nResyn, vTruthRanges);
    cudaDeviceSynchronize();

    int nTruthTableLen = -1;
    cudaMemcpy(&nTruthTableLen, &vTruthRanges[nResyn - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("Truth table len: %d\n", nTruthTableLen);
    cudaMalloc(&vTruth, nTruthTableLen * sizeof(unsigned));
    cudaMalloc(&vTruthElem, cutSize * dUtils::TruthWordNum(cutSize) * sizeof(unsigned));

    // cudaDeviceSynchronize();
    // printTimeStart = hrClock();
    // printMffc<<<1, 1>>>(vCutTable, vCutSizes, vNumSaved, d_pFanin0, d_pFanin1, nNodes, nPIs, nPOs);
    // cudaDeviceSynchronize();
    // printTime += hrClock() - printTimeStart;
    // printf("******print time = %.2lf\n", printTime / (double) NS_PER_SEC);

    // gather all internal nodes in the cone (post order), and compute truth table
    Aig::getElemTruthTable<<<1, 1>>>(vTruthElem, cutSize);
    Aig::getCutTruthTable<CUT_TABLE_SIZE, STACK_SIZE><<<NUM_BLOCKS(nResyn, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
    // getCutTruthTable<<<1, 1>>>(
        d_pFanin0, d_pFanin1, vNumSaved, vResynInd, vCutTable, vCutSizes, 
        vTruth, vTruthRanges, vTruthElem, nResyn, nPIs, cutSize
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Computed cut truth tables\n");


    // build hashtable of the original aig

    HashTable<uint64, uint32> hashTable((int)(nObjs / HT_LOAD_FACTOR));
    uint64 * htKeys = hashTable.get_keys_storage();
    uint32 * htValues = hashTable.get_values_storage();
    int htCapacity = hashTable.get_capacity();

    Aig::buildHashTable<<<NUM_BLOCKS(nNodes, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        d_pFanin0, d_pFanin1, htKeys, htValues, htCapacity, nNodes, nPIs);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Built initial hash table\n");

    // allocate global subgraph table
    // vSubgLinks indicating idx of next row, if one row in vSubgTable is not enough:
    // -1: unvisited, 0: last row, >0: next row idx
    cudaMalloc(&vSubgTable, 2 * nResyn * SUBG_TABLE_SIZE * sizeof(uint64));
    cudaMalloc(&vSubgLinks, 2 * nResyn * sizeof(int));
    cudaMalloc(&vSubgLens, nResyn * sizeof(int));
    cudaMalloc(&pSubgTableNext, sizeof(int));
    cudaMemset(vSubgLinks, -1, 2 * nResyn * sizeof(int));
    cudaMemset(vSubgLens, -1, nResyn * sizeof(int));
    cudaMemcpy(pSubgTableNext, &nResyn, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // isop & algebraic factoring
    printf("Start resyn\n");
    auto resynStartTime = hrClock();
    // resynCut<<<1, 1>>>(
    resynCut<<<NUM_BLOCKS(nResyn, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vResynInd, vCutTable, vCutSizes, vNumSaved, 
        htKeys, htValues, htCapacity, d_pLevel, 
        vSubgTable, vSubgLinks, vSubgLens, pSubgTableNext,
        vTruth, vTruthRanges, vTruthElem, cutSize, nResyn
    );
    gpuChkStackOverflow( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Finished resyn, time = %.2lf secs\n", (hrClock() - resynStartTime) / (double) NS_PER_SEC);


    // // replace scheme

    // 2. check matching status of nodes in each subgraph in topo order
    // 3. for unmatched nodes, add them into the hash table (do not overlap this with 2);
    //    retreive immediately and update matching status if value is not the same (i.e. node created in other subgraphs)
    printf("Starting insert chioce graphs ...\n");
    auto insertSubgStartTime = hrClock();
    nNewObjs = nObjs;
    cudaMalloc(&vChoicesLit, nResyn * sizeof(int));
    cudaMemset(vChoicesLit, -1, nResyn * sizeof(int));
    // insertChoiceGraphs<<<1, 1>>>(
    insertChoiceGraphs<<<NUM_BLOCKS(nResyn, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vSubgTable, vSubgLinks, vSubgLens, vResynInd, vCutTable, vCutSizes, 
        htKeys, htValues, htCapacity, vChoicesLit, nResyn, nObjs, nPIs, &nNewObjs
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Finished insert chioce graphs, obj counter increased to %d, time = %.2lf secs\n", nNewObjs, 
           (hrClock() - insertSubgStartTime) / (double) NS_PER_SEC);
    
    // 4. retrieve all with sorted values (ids)
    cudaFree(vTruth);
    cudaFree(vTruthRanges);
    cudaFree(vTruthElem);
    cudaFree(vSubgTable);
    cudaFree(vSubgLinks);
    cudaFree(vSubgLens);
    cudaFree(pSubgTableNext);

    cudaMalloc(&vReconstructedKeys, nNewObjs * sizeof(uint64));
    cudaMalloc(&vReconstructedIds, nNewObjs * sizeof(uint32));

    cudaMalloc(&vFanin0New, nNewObjs * sizeof(int));
    cudaMalloc(&vFanin1New, nNewObjs * sizeof(int));
    cudaMemset(vFanin0New + nObjs, -1, (nNewObjs - nObjs) * sizeof(int));
    cudaMemset(vFanin1New + nObjs, -1, (nNewObjs - nObjs) * sizeof(int));
    // the old part of fanins should be exactly the same
    cudaMemcpy(vFanin0New, d_pFanin0, nObjs * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vFanin1New, d_pFanin1, nObjs * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    
    // nNewObjs is updated to be the exact number of objs in the hashtable
    int nBufferLen = nNewObjs;
    auto dumpHashtableStartTime = hrClock();
    nNewObjs = hashTable.retrieve_all(vReconstructedKeys, vReconstructedIds, nNewObjs, 1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    assert(nNewObjs - nNodes > 0);
    printf("Finished retrieval from hashtable, original nNodes = %d, new nNodes = %d\n", nNodes, nNewObjs);

    // unbind all keys to fanins
    unbindAllKeys<<<NUM_BLOCKS(nNewObjs - nNodes, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vReconstructedKeys + nNodes, vReconstructedIds + nNodes, 
        vFanin0New + nObjs, vFanin1New + nObjs, nNewObjs - nNodes, nObjs, nBufferLen
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Finished dumping hashtable and unbind keys, time = %.2lf secs\n", 
           (hrClock() - dumpHashtableStartTime) / (double) NS_PER_SEC);


    // perform choice and remove deleted nodes sequentially
    printf("Start sequential choice and reorder\n");
    auto reorderStartTime = hrClock();
    auto [nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nLevelsNew] = choiceReorder(
        vFanin0New, vFanin1New, vResynInd, vChoicesLit, vCutTable, vCutSizes, 
        pNumFanouts, pOuts, nPIs, nPOs, nObjs, nBufferLen, nResyn, fUseZeros
    );
    printf("Finished sequential choice and reorder, time = %.2lf secs\n", (hrClock() - reorderStartTime) / (double) NS_PER_SEC);

    cudaFree(vCutTable);
    cudaFree(vCutSizes);
    cudaFree(vNumSaved);
    cudaFree(vIdxSeq);
    cudaFree(vResynInd);
    cudaFree(vReconvInd);
    cudaFree(vChoicesLit);
    cudaFree(vFanin0New);
    cudaFree(vFanin1New);
    cudaFree(vReconstructedKeys);
    cudaFree(vReconstructedIds);

    printf("-- Overall runtime: %.2lf secs\n", (hrClock() - startTime) / (double) NS_PER_SEC);

    return {nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nLevelsNew};
}

void updateLevel(int * pLevel, int * pFanin0, int * pFanin1, int nObjs, int nPIs) {
    size_t fanin0Idx, fanin1Idx;
    
    for (int i = 0; i <= nPIs; i++)
        pLevel[i] = 0;
    for (int i = nPIs + 1; i < nObjs; i++) {
        fanin0Idx = (size_t)(pFanin0[i] >> 1);
        fanin1Idx = (size_t)(pFanin1[i] >> 1);
        assert(fanin0Idx < i && fanin1Idx < i);

        pLevel[i] = 1 + max(pLevel[fanin0Idx], pLevel[fanin1Idx]);
    }
}

void AIGMan::refactor(bool fAlgMFFC, bool fUseZeros, int cutSize) {
    bool fDeduplicate = false;

    if (fUseZeros)
        printf("refactor: use zeros activated!\n");
    
    if (fAlgMFFC)
        printf("refactor: perform MFFC covering algorithm\n");

    if (cutSize > MAX_CUT_SIZE) {
        printf("refactor: maximum cut size is %d!\n", MAX_CUT_SIZE);
        return;
    }
    if (cutSize != 12)
        printf("refactor: cut size updated to %d\n", cutSize);
    
    if (!aigCreated) {
        printf("refactor: AIG is null! \n");
        return;
    }

double startFullTime = hrClock();

    // make sure data is on host since we need to compute level info
    if (deviceAllocated) {
        toHost();
        clearDevice();
    }
    // allocate and compute level info
    pLevel = (int *) malloc(nObjs * sizeof(int));
    updateLevel(pLevel, pFanin0, pFanin1, nObjs, nPIs);
    // copy host data to device
    cudaMalloc(&d_pLevel, nObjs * sizeof(int));
    cudaMemcpy(d_pLevel, pLevel, nObjs * sizeof(int), cudaMemcpyHostToDevice);
    toDevice();

    // enlarge cuda call stack size
    size_t cuStackSize = 0;
    size_t cuStackSizeEnlarged = 65536;
    cudaDeviceGetLimit(&cuStackSize, cudaLimitStackSize);
    printf("Refactor: setting cudaLimitStackSize = %lu\n", cuStackSizeEnlarged);
    cudaDeviceSetLimit(cudaLimitStackSize, cuStackSizeEnlarged);
    cudaDeviceGetLimit(&cuStackSizeEnlarged, cudaLimitStackSize);
    printf("Refactor: checked cudaLimitStackSize = %lu\n", cuStackSizeEnlarged);

    int nObjsNew, nLevelsNew;
    int * vhFanin0New, * vhFanin1New, * vhOutsNew;
double startAlgTime = hrClock();
    if (fAlgMFFC) {
        std::tie(nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nLevelsNew) = refactorMFFCPerform(
            fUseZeros, cutSize, nObjs, nPIs, nPOs, nNodes, 
            d_pFanin0, d_pFanin1, d_pOuts, d_pNumFanouts, d_pLevel, pOuts, pNumFanouts);
    } else {
        std::tie(nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nLevelsNew) = refactorPerform(
            fUseZeros, cutSize, nObjs, nPIs, nPOs, nNodes, 
            d_pFanin0, d_pFanin1, d_pOuts, d_pNumFanouts, d_pLevel, pOuts, pNumFanouts);
    }
    nLevels = nLevelsNew;
prevAlgTime = hrClock() - startAlgTime;
totalAlgTime += prevAlgTime;


    // free device data, pLevel and d_pLevel
    clearDevice();
    cudaFree(d_pLevel);
    free(pLevel);
    cudaDeviceSynchronize();

    // set back the call stack size
    cudaDeviceSetLimit(cudaLimitStackSize, cuStackSize);

    if (nObjsNew == -1) {
        // input AIG not changed
        printf("input AIG is not changed, since no cut meets the resyn requirements\n");
        return;
    }

    // update host data
    nObjs = nObjsNew;
    nNodes = nObjsNew - nPIs - 1;
    free(pFanin0);
    free(pFanin1);
    free(pOuts);
    free(pNumFanouts);
    pFanin0 = vhFanin0New;
    pFanin1 = vhFanin1New;
    pOuts = vhOutsNew;

    pNumFanouts = (int *) calloc(nObjs, sizeof(int));

    // compute num fanouts
    for (int i = 0; i < nNodes; i++) {
        // literal of LHS (2x variable idx)
        size_t thisIdx = (size_t)(i + 1 + nPIs);

        ++pNumFanouts[AigNodeID(pFanin0[thisIdx])];
        ++pNumFanouts[AigNodeID(pFanin1[thisIdx])];
    }
    for (int i = 0; i < nPOs; ++i)
        ++pNumFanouts[AigNodeID(pOuts[i])];

    if (fDeduplicate) {
        int * vFanin0New, * vFanin1New, * vOutsNew, * vNumFanoutsNew;
        int levelCount;
        toDevice();
        std::tie(nObjsNew, vFanin0New, vFanin1New, vOutsNew, vNumFanoutsNew, levelCount) = Aig::strash(
            d_pFanin0, d_pFanin1, d_pOuts, d_pNumFanouts, nObjs, nPIs, nPOs
        );

        nObjs = nObjsNew;
        nNodes = nObjsNew - nPIs - 1;
        nLevels = levelCount;
        cudaMemcpy(d_pnObjs, &nObjs, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pnNodes, &nNodes, sizeof(int), cudaMemcpyHostToDevice);
        cudaFree(d_pFanin0);
        cudaFree(d_pFanin1);
        cudaFree(d_pOuts);
        cudaFree(d_pNumFanouts);
        d_pFanin0 = vFanin0New;
        d_pFanin1 = vFanin1New;
        d_pOuts = vOutsNew;
        d_pNumFanouts = vNumFanoutsNew;

        assert(deviceAllocated); // on device
    } else {
        assert(!deviceAllocated); // on host
    }

    prevCmdRewrite = 0;
prevFullTime = hrClock() - startFullTime;
totalFullTime += prevFullTime;
}
