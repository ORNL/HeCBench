#include <ctime>
#include <vector>
#include <algorithm>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include "robin_hood.h"
#include "aig_manager.h"
#include "strash.cuh"
#include "hash_table.h"
#include "print.cuh"

/**
 * Create a hashtable containing all the nodes given by pFanin0/1.
 * Assume that pFanin0/1 is already strashed, i.e., no duplicate node and topo order,
 * since the trivial cases are not checked during insertion.
 **/
__global__ void Aig::buildHashTable(const int * pFanin0, const int * pFanin1, 
                                    uint64 * htKeys, uint32 * htValues, int htCapacity,
                                    int nNodes, int nPIs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lit0, lit1, temp;
    if (idx < nNodes) {
        uint32 id = idx + nPIs + 1;
        lit0 = pFanin0[id], lit1 = pFanin1[id];
        if (lit0 > lit1)
            temp = lit0, lit0 = lit1, lit1 = temp;
        uint64 key = formAndNodeKey(lit0, lit1);
        insert_single_no_update<uint64, uint32>(htKeys, htValues, key, id, htCapacity);
    }
}


__global__ void markReadyNodes(const int * pFanin0, const int * pFanin1,
                               const int * vRemainNodes, const int * vOld2NewLit, int * vMarks, int nRemain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nodeId;
    int id0, id1;
    if (idx < nRemain) {
        nodeId = vRemainNodes[idx];
        id0 = dUtils::AigNodeID(pFanin0[nodeId]), id1 = dUtils::AigNodeID(pFanin1[nodeId]);
        if (vOld2NewLit[id0] != -1 && vOld2NewLit[id1] != -1) {
            // both of its two fanins are already reconstructed
            vMarks[idx] = 1;
        }
    }
}

__global__ void insertLevelNodes(const int * vReadyNodes, const int * vOld2NewLit,
                                 const int * pFanin0, const int * pFanin1,
                                 uint64 * htKeys, uint32 * htValues, int htCapacity,
                                 uint64 * vKeysBuffer, int idCounter, int nReady) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nodeId;
    int lit0, lit1, id0, id1, temp;
    uint32 temp0;
    uint64 key;
    if (idx < nReady) {
        nodeId = vReadyNodes[idx];
        
        lit0 = pFanin0[nodeId], lit1 = pFanin1[nodeId]; // old lits
        id0 = dUtils::AigNodeID(lit0), id1 = dUtils::AigNodeID(lit1); // old ids
        // convert to new lit
        lit0 = dUtils::AigNodeNotCond(vOld2NewLit[id0], dUtils::AigNodeIsComplement(lit0));
        lit1 = dUtils::AigNodeNotCond(vOld2NewLit[id1], dUtils::AigNodeIsComplement(lit1));
        if (lit0 > lit1)
            temp = lit0, lit0 = lit1, lit1 = temp;
        
        assert(dUtils::AigNodeID(lit0) < idCounter + idx);
        assert(dUtils::AigNodeID(lit1) < idCounter + idx);
        
        key = formAndNodeKey(lit0, lit1);
        vKeysBuffer[idx] = key;
        
        // check trivial
        temp0 = checkTrivialAndCases(lit0, lit1);
        if (temp0 == HASHTABLE_EMPTY_VALUE<uint64, uint32>) {
            // non-trivial, insert into hashtable
            // assign new (tentative) id as idCounter + idx, which is unique
            insert_single_no_update<uint64, uint32>(htKeys, htValues, key, 
                                                    (uint32)(idCounter + idx), htCapacity);
        }
    }
}

__global__ void updateLevelNodesNewIds(const int * vReadyNodes, int * vOld2NewLit, const uint64 * vKeysBuffer,
                                       const uint64 * htKeys, const uint32 * htValues, int htCapacity,
                                       int nReady) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nodeId;
    uint32 lit0, lit1;
    uint32 temp0;
    uint64 key;
    if (idx < nReady) {
        nodeId = vReadyNodes[idx];
        key = vKeysBuffer[idx];
        unbindAndNodeKeys(key, &lit0, &lit1);

        // check trivial
        temp0 = checkTrivialAndCases((int)lit0, (int)lit1);
        if (temp0 == HASHTABLE_EMPTY_VALUE<uint64, uint32>) {
            // non-trivial
            temp0 = retrieve_single<uint64, uint32>(htKeys, htValues, key, htCapacity); // id
            temp0 = temp0 << 1; // convert to lit
        }

        assert(vOld2NewLit[nodeId] == -1);
        vOld2NewLit[nodeId] = temp0;
    }
}

__global__ void assignOld2NewConsecutiveIds(const uint32 * vOldIds, int * vOld2NewIdConsecutive, 
                                            int nPIs, int nObjsNew) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nPIs + 1) {
        vOld2NewIdConsecutive[idx] = idx;
    } else if (idx < nObjsNew) {
        uint32 nodeIdOld = vOldIds[idx - nPIs - 1];
        assert(nodeIdOld > (uint32)nPIs);
        vOld2NewIdConsecutive[nodeIdOld] = idx;
    }
}

__global__ void unbindKeysReId(const uint64 * vOldKeys, const int * vOld2NewIdConsecutive,
                               int * vFanin0New, int * vFanin1New, int * vNumFanoutsNew,
                               int nNodesNew) {
    // NOTE vFanin0New, vFanin1New should point to the begin of AND node storage;
    //      vNumFanoutsNew should point to the begin of PIs + AND node storage!
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nNodesNew) {
        uint32 oldLit0, oldLit1;
        int oldId0, oldId1, newId0, newId1;
        unbindAndNodeKeys(vOldKeys[idx], &oldLit0, &oldLit1);

        oldId0 = dUtils::AigNodeID((int)oldLit0), oldId1 = dUtils::AigNodeID((int)oldLit1);
        newId0 = vOld2NewIdConsecutive[oldId0], newId1 = vOld2NewIdConsecutive[oldId1];

        vFanin0New[idx] = dUtils::AigNodeLitCond(newId0, dUtils::AigNodeIsComplement((int)oldLit0));
        vFanin1New[idx] = dUtils::AigNodeLitCond(newId1, dUtils::AigNodeIsComplement((int)oldLit1));
        atomicAdd(&vNumFanoutsNew[newId0], 1);
        atomicAdd(&vNumFanoutsNew[newId1], 1);
    }
}

__global__ void poReId(const int * vOld2NewIdConsecutive,
                       int * vOutsNew, int * vNumFanoutsNew, int nPOs) {
    // strashed id -> consecutive id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPOs) {
        int oldLit, oldId, newId;
        
        oldLit = vOutsNew[idx];
        oldId = dUtils::AigNodeID(oldLit);
        newId = vOld2NewIdConsecutive[oldId];

        vOutsNew[idx] = dUtils::AigNodeLitCond(newId, dUtils::AigNodeIsComplement(oldLit));
        atomicAdd(&vNumFanoutsNew[newId], 1);
    }
}

__global__ void poUpdateId(const int * pOuts, const int * vOld2NewLit,
                           int * vOutsNew, int nPOs) {
    // old id -> strashed id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPOs) {
        int oldLit, oldId;

        oldLit = pOuts[idx];
        oldId = dUtils::AigNodeID(oldLit);
        vOutsNew[idx] = dUtils::AigNodeNotCond(vOld2NewLit[oldId], dUtils::AigNodeIsComplement(oldLit));
    }
}

__global__ void markDanglingNodesIter(const int * pFanin0, const int * pFanin1, int * pNumFanouts,
                                      int * vDanglingMarks, int nNodes, int nPIs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nNodes) {
        int nodeId = idx + nPIs + 1, id0, id1;
        assert(pNumFanouts[nodeId] >= 0);
        if (!vDanglingMarks[idx] && pNumFanouts[nodeId] == 0) {
            vDanglingMarks[idx] = 1;
            id0 = dUtils::AigNodeID(pFanin0[nodeId]), id1 = dUtils::AigNodeID(pFanin1[nodeId]);
            atomicSub(&pNumFanouts[id0], 1);
            atomicSub(&pNumFanouts[id1], 1);
        }
    }
}

std::tuple<int, int *, int *, int *, int *, int>
Aig::strash(const int * pFanin0, const int * pFanin1, const int * pOuts, int * pNumFanouts,
            int nObjs, int nPIs, int nPOs) {
    int * vRemainNodes, * vOld2NewLit;
    int * vReadyNodes, * vMarks;
    uint64 * vKeysBuffer;
    uint32 * vValuesBuffer;
    int * vFanin0New, * vFanin1New, * vOutsNew, * vNumFanoutsNew;

    int * pNewGlobalListEnd;

    int nNodes = nObjs - nPIs - 1;
    int nRemain, nReady;

    printf("GPU strash: start with nNodes = %d\n", nNodes);
    auto startTime = hrClock();

    HashTable<uint64, uint32> hashTable(nObjs * 2);
    uint64 * htKeys = hashTable.get_keys_storage();
    uint32 * htValues = hashTable.get_values_storage();
    int htCapacity = hashTable.get_capacity();

    hipMalloc(&vRemainNodes, nNodes * sizeof(int));
    hipMalloc(&vReadyNodes, nNodes * sizeof(int));
    hipMalloc(&vMarks, nNodes * sizeof(int));
    hipMalloc(&vOld2NewLit, nObjs * sizeof(int));
    hipMalloc(&vKeysBuffer, nNodes * sizeof(uint64));
    hipMalloc(&vValuesBuffer, nNodes * sizeof(uint32));
    hipMemset(vOld2NewLit, -1, nObjs * sizeof(int));
    hipMemset(vMarks, 0, nNodes * sizeof(int));

    // mark dangling nodes first
    int nDangling, nDanglingNew = 0;
    do {
        nDangling = nDanglingNew;
        markDanglingNodesIter<<<NUM_BLOCKS(nNodes, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            pFanin0, pFanin1, pNumFanouts, vMarks, nNodes, nPIs
        );
        nDanglingNew = thrust::reduce(thrust::device, vMarks, vMarks + nNodes);
    } while (nDanglingNew != nDangling);
    printf("  detected %d dangling nodes\n", nDangling);

    // generate remaining node list
    thrust::sequence(thrust::device, vRemainNodes, vRemainNodes + nNodes, nPIs + 1);
    nRemain = nNodes;
    if (nDangling > 0) {
        pNewGlobalListEnd = thrust::remove_if(
            thrust::device, vRemainNodes, vRemainNodes + nNodes, 
            vMarks, thrust::identity<int>()
        );
        nRemain = pNewGlobalListEnd - vRemainNodes;
        assert(nRemain + nDangling == nNodes);
    }
    thrust::sequence(thrust::device, vOld2NewLit, vOld2NewLit + nPIs + 1, 0, 2); // lits for PIs does not change

    int idCounter = nPIs + 1;
    int levelCount = 0;
    while (nRemain > 0) {
        hipMemset(vMarks, 0, nRemain * sizeof(int));
        markReadyNodes<<<NUM_BLOCKS(nRemain, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            pFanin0, pFanin1, vRemainNodes, vOld2NewLit, vMarks, nRemain
        );

        pNewGlobalListEnd = thrust::copy_if(
            thrust::device, vRemainNodes, vRemainNodes + nRemain, 
            vMarks, vReadyNodes, thrust::identity<int>()
        );
        nReady = pNewGlobalListEnd - vReadyNodes;

        pNewGlobalListEnd = thrust::remove_if(
            thrust::device, vRemainNodes, vRemainNodes + nRemain, 
            vMarks, thrust::identity<int>()
        );
        assert((pNewGlobalListEnd - vRemainNodes) + nReady == nRemain);
        nRemain = pNewGlobalListEnd - vRemainNodes;

        insertLevelNodes<<<NUM_BLOCKS(nReady, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vReadyNodes, vOld2NewLit, pFanin0, pFanin1, htKeys, htValues, htCapacity,
            vKeysBuffer, idCounter, nReady
        );
        updateLevelNodesNewIds<<<NUM_BLOCKS(nReady, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vReadyNodes, vOld2NewLit, vKeysBuffer, htKeys, htValues, htCapacity, nReady
        );
        // increment idCounter
        assert(idCounter + nReady < (INT_MAX / 2));
        idCounter += nReady;
        levelCount++;
    }

    int nNodesNew = hashTable.retrieve_all(vKeysBuffer, vValuesBuffer, nNodes, 1);
    int nObjsNew = nNodesNew + nPIs + 1;
    int * vOld2NewIdConsecutive = vOld2NewLit; // map old id to new consecutive ids
    hipMalloc(&vFanin0New, nObjsNew * sizeof(int));
    hipMalloc(&vFanin1New, nObjsNew * sizeof(int));
    hipMalloc(&vOutsNew, nPOs * sizeof(int));
    hipMalloc(&vNumFanoutsNew, nObjsNew * sizeof(int));
    hipMemset(vNumFanoutsNew, 0, nObjsNew * sizeof(int));

    poUpdateId<<<NUM_BLOCKS(nPOs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        pOuts, vOld2NewLit, vOutsNew, nPOs
    );
    assignOld2NewConsecutiveIds<<<NUM_BLOCKS(nObjsNew, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vValuesBuffer, vOld2NewIdConsecutive, nPIs, nObjsNew
    );
    unbindKeysReId<<<NUM_BLOCKS(nNodesNew, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vKeysBuffer, vOld2NewIdConsecutive, 
        vFanin0New + nPIs + 1, vFanin1New + nPIs + 1, vNumFanoutsNew, nNodesNew
    );
    poReId<<<NUM_BLOCKS(nPOs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vOld2NewIdConsecutive, vOutsNew, vNumFanoutsNew, nPOs
    );
    hipDeviceSynchronize();

    hipFree(vRemainNodes);
    hipFree(vReadyNodes);
    hipFree(vMarks);
    hipFree(vOld2NewLit);
    hipFree(vKeysBuffer);
    hipFree(vValuesBuffer);

    printf("GPU strash: finish with nNodes = %d, time = %.2lf sec\n", nNodesNew,
           (hrClock() - startTime) / (double) NS_PER_SEC);

    return {nObjsNew, vFanin0New, vFanin1New, vOutsNew, vNumFanoutsNew, levelCount};
}


void AIGMan::strash(bool fCPU, bool fRecordTime) {

double startFullTime = hrClock();
    if (fCPU) {
        // CPU strash
        if (deviceAllocated) {
            toHost();
            clearDevice();
        }

double startAlgTime = hrClock();

        int * pFanin0New, * pFanin1New, * pOutsNew, * pNumFanoutsNew;
        int * vDanglingMarks;
        pFanin0New = (int *) malloc(nObjs * sizeof(int));
        pFanin1New = (int *) malloc(nObjs * sizeof(int));
        pOutsNew = (int *) malloc(nPOs * sizeof(int));
        pNumFanoutsNew = (int *) calloc(nObjs, sizeof(int));
        vDanglingMarks = (int *) calloc(nObjs, sizeof(int));

        printf("CPU strash: start with nNodes = %d\n", nNodes);

        int lit0, lit1, id0, id1, idCounter, temp, danglingCounter;
        uint64 key;
        robin_hood::unordered_map<uint64, int> strashTable;
        std::vector<int> oldToNew(nObjs, -1);
        strashTable.reserve(nNodes);

        std::function<uint64(int, int)> formKey = [](int lit0, int lit1){
            assert(lit0 <= lit1);
            uint32 uLit0 = (uint32)lit0;
            uint32 uLit1 = (uint32)lit1;
            return ((uint64)uLit0) << 32 | uLit1;
        };

        std::function<void(int, int *, int *, int *, int *, int)> markDeleteNode = 
        [&](int nodeId, int * vFanin0, int * vFanin1, 
           int * vNumFanouts, int * vDanglingMarks,
           int numPIs) {
            assert(vNumFanouts[nodeId] == 0);
            if (dUtils::AigIsPIConst(nodeId, numPIs)) return;
            if (vDanglingMarks[nodeId]) return;

            int faninId0, faninId1;
            vDanglingMarks[nodeId] = 1;
            faninId0 = AigNodeID(vFanin0[nodeId]), faninId1 = AigNodeID(vFanin1[nodeId]);
            vNumFanouts[faninId0]--;
            vNumFanouts[faninId1]--;

            assert(vNumFanouts[faninId0] >= 0 && vNumFanouts[faninId1] >= 0);
            if (vNumFanouts[faninId0] == 0)
                markDeleteNode(faninId0, vFanin0, vFanin1, vNumFanouts, vDanglingMarks, numPIs);
            if (vNumFanouts[faninId1] == 0)
                markDeleteNode(faninId1, vFanin0, vFanin1, vNumFanouts, vDanglingMarks, numPIs);
        };


        // cleanup dangling nodes by marking them
        for (int i = 0; i < nNodes; i++) {
            int thisIdx = i + 1 + nPIs;
            if (pNumFanouts[thisIdx] == 0)
                markDeleteNode(thisIdx, pFanin0, pFanin1, pNumFanouts, vDanglingMarks, nPIs);
        }


        // initialize oldToNew for PIs
        for (int i = 0; i <= nPIs; i++)
            oldToNew[i] = i;
        
        // initialize delay info
        std::vector<int> vDelays(nObjs, -1);
        for (int i = 0; i <= nPIs; i++)
            vDelays[i] = 0;

        idCounter = 1 + nPIs;
        danglingCounter = 0;
        int maxDelay = -1;
        for (int i = 0; i < nNodes; i++) {
            int thisIdx = i + 1 + nPIs;
            if (vDanglingMarks[thisIdx]) {
                danglingCounter++;
                continue;
            }

            lit0 = pFanin0[thisIdx], lit1 = pFanin1[thisIdx];
            id0 = AigNodeID(lit0), id1 = AigNodeID(lit1);
            
            // map to new id
            lit0 = dUtils::AigNodeLitCond(oldToNew[id0], dUtils::AigNodeIsComplement(lit0));
            lit1 = dUtils::AigNodeLitCond(oldToNew[id1], dUtils::AigNodeIsComplement(lit1));
            if (lit0 > lit1) {
                temp = lit0, lit0 = lit1, lit1 = temp;
            }

            key = formKey(lit0, lit1);

            // strashing
            auto strashRet = strashTable.find(key);
            if (strashRet == strashTable.end()) {
                // new node, insert
                strashTable[key] = idCounter;
                oldToNew[thisIdx] = idCounter;
                // save results
                pFanin0New[idCounter] = lit0;
                pFanin1New[idCounter] = lit1;
                ++pNumFanoutsNew[AigNodeID(lit0)];
                ++pNumFanoutsNew[AigNodeID(lit1)];
                
                assert(vDelays[AigNodeID(lit0)] != -1 && vDelays[AigNodeID(lit1)] != -1);
                vDelays[idCounter] = 1 + std::max(vDelays[AigNodeID(lit0)], vDelays[AigNodeID(lit1)]);
                maxDelay = std::max(maxDelay, vDelays[idCounter]);
                
                ++idCounter;
            } else {
                // already exists, retrieve
                oldToNew[thisIdx] = strashRet->second;
            }
        }

        for (int i = 0; i < nPOs; i++) {
            lit0 = pOuts[i];
            id0 = AigNodeID(lit0);
            lit0 = dUtils::AigNodeLitCond(oldToNew[id0], dUtils::AigNodeIsComplement(lit0));

            pOutsNew[i] = lit0;
            ++pNumFanoutsNew[AigNodeID(lit0)];
        }
if (fRecordTime) {
    prevAlgTime = hrClock() - startAlgTime;
    totalAlgTime += prevAlgTime;
}

        nObjs = idCounter;
        nNodes = nObjs - nPIs - 1;
        nLevels = maxDelay;
        free(pFanin0);
        free(pFanin1);
        free(pOuts);
        free(pNumFanouts);
        free(vDanglingMarks);
        pFanin0 = pFanin0New, pFanin1 = pFanin1New, pOuts = pOutsNew, pNumFanouts = pNumFanoutsNew;

        if (danglingCounter > 0)
            printf("  removed %d dangling nodes\n", danglingCounter);
        printf("CPU strash: finish with nNodes = %d\n", nNodes);

        assert(!deviceAllocated); // on host
    } else {
        // GPU strash
        if (!deviceAllocated)
            toDevice();

double startAlgTime = hrClock();
        auto [nObjsNew, vFanin0New, vFanin1New, vOutsNew, vNumFanoutsNew, levelCount] = Aig::strash(
            d_pFanin0, d_pFanin1, d_pOuts, d_pNumFanouts, nObjs, nPIs, nPOs
        );
if (fRecordTime) {
    prevAlgTime = hrClock() - startAlgTime;
    totalAlgTime += prevAlgTime;
}

        nObjs = nObjsNew;
        nNodes = nObjsNew - nPIs - 1;
        nLevels = levelCount;
        hipMemcpy(d_pnObjs, &nObjs, sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_pnNodes, &nNodes, sizeof(int), hipMemcpyHostToDevice);
        hipFree(d_pFanin0);
        hipFree(d_pFanin1);
        hipFree(d_pOuts);
        hipFree(d_pNumFanouts);
        d_pFanin0 = vFanin0New;
        d_pFanin1 = vFanin1New;
        d_pOuts = vOutsNew;
        d_pNumFanouts = vNumFanoutsNew;

        assert(deviceAllocated); // on device
    }
    
    prevCmdRewrite = 0;
if (fRecordTime) {
    prevFullTime = hrClock() - startFullTime;
    totalFullTime += prevFullTime;
}
}
