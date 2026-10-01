#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <ctime>
#include <vector>
#include <algorithm>

#include "robin_hood.h"
#include "aig_manager.h"
#include "strash.hpp"
#include "hash_table.h"
#include "print.hpp"
#include <time.h>

#include <cmath>

struct identity {
    template <typename T>
    T operator()(const T& x) const { return x; }
};

/**
 * Create a hashtable containing all the nodes given by pFanin0/1.
 * Assume that pFanin0/1 is already strashed, i.e., no duplicate node and topo order,
 * since the trivial cases are not checked during insertion.
 **/
SYCL_EXTERNAL void Aig::buildHashTable(const int *pFanin0, const int *pFanin1,
                                       uint64 *htKeys, uint32 *htValues,
                                       int htCapacity, int nNodes, int nPIs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
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


void markReadyNodes(const int * pFanin0, const int * pFanin1,
                               const int * vRemainNodes, const int * vOld2NewLit, int * vMarks, int nRemain) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
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

// Single-pass atomic partition of the remaining-node list into "ready this
// level" (both fanins already reconstructed) and "still remaining". Replaces
// the old markReadyNodes + oneDPL copy_if + remove_if trio, cutting the per
// level host syncs from ~3 to 1 and removing per-level oneDPL temp allocations.
// This matters a lot for deep circuits (hyp has ~25k topological levels).
// Order within a level is irrelevant to correctness (new ids stay > fanin ids).
void partitionReadyNodes(const int *pFanin0, const int *pFanin1,
                         const int *vRemainNodes, const int *vOld2NewLit,
                         int *vReadyOut, int *vRemainOut, int *dCounts,
                         int nRemain) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nRemain) {
        int nodeId = vRemainNodes[idx];
        int id0 = dUtils::AigNodeID(pFanin0[nodeId]);
        int id1 = dUtils::AigNodeID(pFanin1[nodeId]);
        if (vOld2NewLit[id0] != -1 && vOld2NewLit[id1] != -1) {
            int pos = nsycl::atomic_fetch_add(&dCounts[0], 1);
            vReadyOut[pos] = nodeId;
        } else {
            int pos = nsycl::atomic_fetch_add(&dCounts[1], 1);
            vRemainOut[pos] = nodeId;
        }
    }
}

void insertLevelNodes(const int * vReadyNodes, const int * vOld2NewLit,
                                 const int * pFanin0, const int * pFanin1,
                                 uint64 * htKeys, uint32 * htValues, int htCapacity,
                                 uint64 * vKeysBuffer, int idCounter, int nReady) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
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

void updateLevelNodesNewIds(const int * vReadyNodes, int * vOld2NewLit, const uint64 * vKeysBuffer,
                                       const uint64 * htKeys, const uint32 * htValues, int htCapacity,
                                       int nReady) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
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

void assignOld2NewConsecutiveIds(const uint32 * vOldIds, int * vOld2NewIdConsecutive, 
                                            int nPIs, int nObjsNew) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx <= nPIs + 1) {
        vOld2NewIdConsecutive[idx] = idx;
    } else if (idx < nObjsNew) {
        uint32 nodeIdOld = vOldIds[idx - nPIs - 1];
        assert(nodeIdOld > (uint32)nPIs);
        vOld2NewIdConsecutive[nodeIdOld] = idx;
    }
}

void unbindKeysReId(const uint64 * vOldKeys, const int * vOld2NewIdConsecutive,
                               int * vFanin0New, int * vFanin1New, int * vNumFanoutsNew,
                               int nNodesNew) {
    // NOTE vFanin0New, vFanin1New should point to the begin of AND node storage;
    //      vNumFanoutsNew should point to the begin of PIs + AND node storage!
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nNodesNew) {
        uint32 oldLit0, oldLit1;
        int oldId0, oldId1, newId0, newId1;
        unbindAndNodeKeys(vOldKeys[idx], &oldLit0, &oldLit1);

        oldId0 = dUtils::AigNodeID((int)oldLit0), oldId1 = dUtils::AigNodeID((int)oldLit1);
        newId0 = vOld2NewIdConsecutive[oldId0], newId1 = vOld2NewIdConsecutive[oldId1];

        vFanin0New[idx] = dUtils::AigNodeLitCond(newId0, dUtils::AigNodeIsComplement((int)oldLit0));
        vFanin1New[idx] = dUtils::AigNodeLitCond(newId1, dUtils::AigNodeIsComplement((int)oldLit1));
        nsycl::atomic_fetch_add(
            &vNumFanoutsNew[newId0], 1);
        nsycl::atomic_fetch_add(
            &vNumFanoutsNew[newId1], 1);
    }
}

void poReId(const int * vOld2NewIdConsecutive,
                       int * vOutsNew, int * vNumFanoutsNew, int nPOs) {
    // strashed id -> consecutive id
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nPOs) {
        int oldLit, oldId, newId;
        
        oldLit = vOutsNew[idx];
        oldId = dUtils::AigNodeID(oldLit);
        newId = vOld2NewIdConsecutive[oldId];

        vOutsNew[idx] = dUtils::AigNodeLitCond(newId, dUtils::AigNodeIsComplement(oldLit));
        nsycl::atomic_fetch_add(
            &vNumFanoutsNew[newId], 1);
    }
}

void poUpdateId(const int * pOuts, const int * vOld2NewLit,
                           int * vOutsNew, int nPOs) {
    // old id -> strashed id
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nPOs) {
        int oldLit, oldId;

        oldLit = pOuts[idx];
        oldId = dUtils::AigNodeID(oldLit);
        vOutsNew[idx] = dUtils::AigNodeNotCond(vOld2NewLit[oldId], dUtils::AigNodeIsComplement(oldLit));
    }
}

void markDanglingNodesIter(const int * pFanin0, const int * pFanin1, int * pNumFanouts,
                                      int * vDanglingMarks, int nNodes, int nPIs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nNodes) {
        int nodeId = idx + nPIs + 1, id0, id1;
        assert(pNumFanouts[nodeId] >= 0);
        if (!vDanglingMarks[idx] && pNumFanouts[nodeId] == 0) {
            vDanglingMarks[idx] = 1;
            id0 = dUtils::AigNodeID(pFanin0[nodeId]), id1 = dUtils::AigNodeID(pFanin1[nodeId]);
            nsycl::atomic_fetch_sub(
                &pNumFanouts[id0], 1);
            nsycl::atomic_fetch_sub(
                &pNumFanouts[id1], 1);
        }
    }
}

std::tuple<int, int *, int *, int *, int *, int>
Aig::strash(const int *pFanin0, const int *pFanin1, const int *pOuts,
            int *pNumFanouts, int nObjs, int nPIs, int nPOs) {
    sycl::queue &q_ct1 = nsycl::queue();
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

    vRemainNodes = sycl::malloc_device<int>(nNodes, q_ct1);
    vReadyNodes = sycl::malloc_device<int>(nNodes, q_ct1);
    int *vRemainTmp = sycl::malloc_device<int>(nNodes, q_ct1);
    int *dCounts = sycl::malloc_device<int>(2, q_ct1);
    int hCounts[2];
    vMarks = sycl::malloc_device<int>(nNodes, q_ct1);
    vOld2NewLit = sycl::malloc_device<int>(nObjs, q_ct1);
    vKeysBuffer = sycl::malloc_device<uint64>(nNodes, q_ct1);
    vValuesBuffer = sycl::malloc_device<uint32>(nNodes, q_ct1);
    q_ct1.memset(vOld2NewLit, -1, nObjs * sizeof(int)).wait();
    q_ct1.memset(vMarks, 0, nNodes * sizeof(int)).wait();

    // mark dangling nodes first
    int nDangling, nDanglingNew = 0;
    do {
        nDangling = nDanglingNew;
        q_ct1.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(nNodes, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                markDanglingNodesIter(pFanin0, pFanin1, pNumFanouts, vMarks,
                                      nNodes, nPIs);
            });
        nDanglingNew =
            std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1),
                        vMarks, vMarks + nNodes);
    } while (nDanglingNew != nDangling);
    printf("  detected %d dangling nodes\n", nDangling);

    // generate remaining node list
    nsycl::iota(q_ct1, vRemainNodes,
               vRemainNodes + nNodes, nPIs + 1);
    nRemain = nNodes;
    if (nDangling > 0) {
        pNewGlobalListEnd = nsycl::remove_if(q_ct1, vRemainNodes,
            vRemainNodes + nNodes, vMarks, identity{});
        nRemain = pNewGlobalListEnd - vRemainNodes;
        assert(nRemain + nDangling == nNodes);
    }
    nsycl::iota(q_ct1, vOld2NewLit,
               vOld2NewLit + nPIs + 1, 0, 2); // lits for PIs does not change

    int idCounter = nPIs + 1;
    int levelCount = 0;
    while (nRemain > 0) {
        // reset the two device counters (ready / still-remaining); no host wait
        // needed: the in-order queue serialises this before the kernel below.
        q_ct1.memset(dCounts, 0, 2 * sizeof(int));
        q_ct1.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(nRemain, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                partitionReadyNodes(pFanin0, pFanin1, vRemainNodes, vOld2NewLit,
                                    vReadyNodes, vRemainTmp, dCounts, nRemain);
            });
        // single host sync per level to fetch the two counts
        q_ct1.memcpy(hCounts, dCounts, 2 * sizeof(int)).wait();
        nReady = hCounts[0];
        assert(hCounts[0] + hCounts[1] == nRemain);
        nRemain = hCounts[1];

        q_ct1.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(nReady, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                insertLevelNodes(vReadyNodes, vOld2NewLit, pFanin0, pFanin1,
                                 htKeys, htValues, htCapacity, vKeysBuffer,
                                 idCounter, nReady);
            });
        q_ct1.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(nReady, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                updateLevelNodesNewIds(vReadyNodes, vOld2NewLit, vKeysBuffer,
                                       htKeys, htValues, htCapacity, nReady);
            });
        // increment idCounter
        assert(idCounter + nReady < (INT_MAX / 2));
        idCounter += nReady;
        // the still-remaining nodes are now compacted into vRemainTmp
        std::swap(vRemainNodes, vRemainTmp);
        levelCount++;
    }

    int nNodesNew = hashTable.retrieve_all(vKeysBuffer, vValuesBuffer, nNodes, 1);
    int nObjsNew = nNodesNew + nPIs + 1;
    int * vOld2NewIdConsecutive = vOld2NewLit; // map old id to new consecutive ids
    vFanin0New = sycl::malloc_device<int>(nObjsNew, q_ct1);
    vFanin1New = sycl::malloc_device<int>(nObjsNew, q_ct1);
    vOutsNew = sycl::malloc_device<int>(nPOs, q_ct1);
    vNumFanoutsNew = sycl::malloc_device<int>(nObjsNew, q_ct1);
    q_ct1.memset(vNumFanoutsNew, 0, nObjsNew * sizeof(int)).wait();

    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nPOs, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            poUpdateId(pOuts, vOld2NewLit, vOutsNew, nPOs);
        });
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nObjsNew, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            assignOld2NewConsecutiveIds(vValuesBuffer, vOld2NewIdConsecutive,
                                        nPIs, nObjsNew);
        });
    q_ct1.submit([&](sycl::handler &cgh) {
        auto vFanin0New_nPIs_ct2 = vFanin0New + nPIs + 1;
        auto vFanin1New_nPIs_ct3 = vFanin1New + nPIs + 1;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(nNodesNew, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                unbindKeysReId(vKeysBuffer, vOld2NewIdConsecutive,
                               vFanin0New_nPIs_ct2, vFanin1New_nPIs_ct3,
                               vNumFanoutsNew, nNodesNew);
            });
    });
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nPOs, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            poReId(vOld2NewIdConsecutive, vOutsNew, vNumFanoutsNew, nPOs);
        });
    q_ct1.wait();

    sycl::free(vRemainNodes, q_ct1);
    sycl::free(vReadyNodes, q_ct1);
    sycl::free(vRemainTmp, q_ct1);
    sycl::free(dCounts, q_ct1);
    sycl::free(vMarks, q_ct1);
    sycl::free(vOld2NewLit, q_ct1);
    sycl::free(vKeysBuffer, q_ct1);
    sycl::free(vValuesBuffer, q_ct1);

    printf("GPU strash: finish with nNodes = %d, time = %.2lf sec\n", nNodesNew,
           (hrClock() - startTime) / (double) NS_PER_SEC);

    return {nObjsNew, vFanin0New, vFanin1New, vOutsNew, vNumFanoutsNew, levelCount};
}

void AIGMan::strash(bool fCPU, bool fRecordTime) {
    sycl::queue &q_ct1 = nsycl::queue();

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
        q_ct1.memcpy(d_pnObjs, &nObjs, sizeof(int));
        q_ct1.memcpy(d_pnNodes, &nNodes, sizeof(int)).wait();
        sycl::free(d_pFanin0, q_ct1);
        sycl::free(d_pFanin1, q_ct1);
        sycl::free(d_pOuts, q_ct1);
        sycl::free(d_pNumFanouts, q_ct1);
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
