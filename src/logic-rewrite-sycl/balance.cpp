#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <vector>
#include <unordered_map>
#include <tuple>

#include <functional>

#include <oneapi/dpl/random>

#include "common.h"
#include "hash_table.h"
#include "balance.h"
#include <time.h>

const int MAX_LOCAL_COVER_SIZE = 1024;
const int DFS_COVER_STACK_SIZE = 128;
const int COVER_TABLE_NUM_COLS = 4;
const double HT_LOAD_FACTOR = 0.5;

// managed memory
inline nsycl::shared_scalar<int> isEnd;

// declarations
#ifdef DEBUG
__global__ void printLevelInfo(int * vNodes, int * vNodesFilter, int levelCount, int len, int nPIs, int nPOs);
void printLevelFilterStats(int * vNodesFilter, int levelCount, int len);
__global__ void printArray(const int * array, const int len);
void printResults(const int * resultFanin0, const int * resultFanin1, const int * resultNumFanouts, const int * resultPOs,
                  const int len, const int nPIs, const int nPOs);
__global__ void printLocalReconstructArray(int * vLocalReconstructArrays, int * vLocalReconstructLens, 
                                           const int * vGatheredReadyCovers, const int nReadyCovers,
                                           const int maxCoverLen, int nPIs, int nPOs);
__global__ void printCoverTable(const int * vCoverTable, const int * vCoverTableLinks, const int * vCoverTableLens);
#endif

void checkCoverTravEnd(int * isEnd, int * vInputs, int * vInputsFilter, 
                                  int * vCanonTable, const int nPIs, const int arrayLen) {
    /**
     * This is a full-filtering marking both "PI" and "R" (in-level redundancy). 
     **/

    // isEnd should be 1 when passed in!
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < arrayLen) {
        int thisLit = vInputs[idx];
        int nodeId = dUtils::AigNodeID(thisLit);
        int nodeIsComplement = dUtils::AigNodeIsComplement(thisLit);

        // convert vInputs from literal to nodeId
        vInputs[idx] = nodeId;

        if (dUtils::AigIsNode(nodeId, nPIs)) {
            // has at least one AND node, cannot stop traversal
            *isEnd = 0;
            // 1 indicates non-duplicate AND node
            vInputsFilter[idx] = 1;

            // only the first attempt thread will change vCanonTable[nodeId] to its idx + 1 (> 0),
            // and get old = 0; later threads will not change vCanonTable[nodeId], and get
            // old = idx of first thread + 1
            int old = nsycl::atomic_compare_exchange(
                &vCanonTable[nodeId], 0, idx + 1);
            if (old > 0) {
                // later threads
                vInputsFilter[idx] = -old;
            }
            // first thread keeps vInputsFilter[idx] = 1 unchanged
        } else {
            // vInputs[idx] is PI
            vInputsFilter[idx] = 0;
        }
    }
}









uint32 retrieveAndNode(int lit1, int lit2,
                                  const uint64 * htReconstructKeys, const uint32 * htReconstructValues, const int htReconstructCapacity) {
    // check trivial cases
    // Note that when encountering these cases during cover reconstruction,
    // the hashtable remains unaffected while the number of inputs is reduced by one,
    // with the cover function unchanged.
    uint32 trivialResult = checkTrivialAndCases(lit1, lit2);
    if (trivialResult != HASHTABLE_EMPTY_VALUE<uint64, uint32>)
        return trivialResult;
    
    // make sure lit1 is smaller than lit2
    if (lit1 > lit2) {
        int temp;
        temp = lit1;
        lit1 = lit2;
        lit2 = temp;
    }
    uint64 key = formAndNodeKey(lit1, lit2);

    // check hash table
    uint32 value = retrieve_single<uint64, uint32>(htReconstructKeys, htReconstructValues, key, htReconstructCapacity);
    // note that retrieved is id, so convert to literal
    if (value != HASHTABLE_EMPTY_VALUE<uint64, uint32>)
        value = value << 1;

    // return HASHTABLE_EMPTY_VALUE if not found
    return value;
}

void prepareDataToInsert(int * vLocalReconstructArrays, int * vLocalReconstructLevels, int * vLocalReconstructLens, 
                                    uint64 * vStepNodeKeys, uint32 * vStepNodeValues, int * vStepNodeMask, uint32 * vStepNodeLevels,
                                    const uint64 * htReconstructKeys, const uint32 * htReconstructValues, const int htReconstructCapacity,
                                    const uint32 * htLevelKeys, const uint32 * htLevelValues, const int htLevelCapacity,
                                    const int newIdCounter, const int nReadyCovers, const int nPIs, const int nPOs, const int maxCoverLen,
                                    int * dMaxMask) {
    /**
     * Prepare one batch of AND node to be inserted to hash table. 
     * 
     * Result of this kernel function:
     * vStepNodeMask = 0 (already complete including constant false, invalid entry): vStepNodeKeys, vStepNodeValues, vStepNodeLevels are all invalid
     * vStepNodeMask = 1 (non-existing nodes): vStepNodeKeys, vStepNodeValues, vStepNodeLevels are all valid
     * vStepNodeMask = 2 (existing nodes): vStepNodeValues, vStepNodeLevels are valid, but vStepNodeKeys are invalid
     * */
    // newIdCounter indicates the max id in reconstruct hashtable + 1, i.e., first available new id.

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nReadyCovers) {
        int length = vLocalReconstructLens[idx];
        int localArrayStartIdx = idx * maxCoverLen;
        assert(length >= 0);

        if (length <= 1) {
            // mask zero
            vStepNodeMask[idx] = 0; // 0 as no insert to hash table and reconstruct array
            return;
        }

        // pop out two last nodes
        int lit1 = vLocalReconstructArrays[localArrayStartIdx + length - 1];
        int lit2 = vLocalReconstructArrays[localArrayStartIdx + length - 2];
        int lv1 = vLocalReconstructLevels[localArrayStartIdx + length - 1];
        int lv2 = vLocalReconstructLevels[localArrayStartIdx + length - 2];
        vLocalReconstructLens[idx] -= 2;

        // perform a retrieval from reconstruct hashtable first
        // if found, update its level and mask zero
        uint32 retrieveRes = retrieveAndNode(lit1, lit2, htReconstructKeys, htReconstructValues, htReconstructCapacity);
        if (retrieveRes != HASHTABLE_EMPTY_VALUE<uint64, uint32>) {
            // printf(" * %s%d,%s%d ", dUtils::AigNodeIsComplement(lit1) ? "!" : "", dUtils::AigNodeIDDebug(lit1, nPIs, nPOs),
            //                        dUtils::AigNodeIsComplement(lit2) ? "!" : "", dUtils::AigNodeIDDebug(lit2, nPIs, nPOs));

            // already exist node
            vStepNodeMask[idx] = 2; // 2 as no insert to hash table but to reconstruct array
            nsycl::atomic_fetch_max(
                dMaxMask, 2);
            vStepNodeValues[idx] = retrieveRes; // note that literal is saved here, but for new node entries id is saved

            // note that if retrieveRes is a trivial case, then it might be PI
            int retrievedId = dUtils::AigNodeID(retrieveRes);
            if (dUtils::AigIsNode(retrievedId, nPIs)) {
                retrieveRes = retrieve_single<uint32, uint32>( // retrieve level from level hashtable
                    htLevelKeys, htLevelValues, (uint32)retrievedId, htLevelCapacity
                );
            } else {
                retrieveRes = 0; // PI has zero level
            }
            assert(retrieveRes != (HASHTABLE_EMPTY_VALUE<uint32, uint32>));

            vStepNodeLevels[idx] = retrieveRes;
            return;
        }

        // non-existing nodes
        if (lit1 > lit2) {
            int temp;
            temp = lit1, lit1 = lit2, lit2 = temp;
        }
        
        vStepNodeMask[idx] = 1; // 1 as insert to hash table and to reconstruct array
        nsycl::atomic_fetch_max(
            dMaxMask, 1);
        vStepNodeKeys[idx] = formAndNodeKey(lit1, lit2);
        vStepNodeValues[idx] = (uint32)(newIdCounter + idx); // save new id, instead of literal as for mask = 2 nodes
        vStepNodeLevels[idx] = (uint32)sycl::max(lv1, lv2) + 1;
    }
}


void sharedNodeDrivenPermute(int * vLocalReconstructArrays, int * vLocalReconstructLevels, int * vLocalReconstructLens, 
                                        const uint64 * htReconstructKeys, const uint32 * htReconstructValues, const int htReconstructCapacity,
                                        const int nReadyCovers, const int maxCoverLen) {
    // this can be optional. 
    // In original ABC we tested that without this step, num of node reduction dropped around 2.5%.
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nReadyCovers) {
        int leftMostIdx, rightBoundIdx;
        int length = vLocalReconstructLens[idx];
        int localArrayStartIdx = idx * maxCoverLen;
        int borderLevel, currLevel;

        if (length <= 1)
            return;

        // part 1. find left most node with same level as right border one

        if (length < 3) {
            leftMostIdx = 0;
        } else {
            leftMostIdx = length - 2;
            borderLevel = vLocalReconstructLevels[localArrayStartIdx + leftMostIdx];

            for (leftMostIdx--; leftMostIdx >= 0; leftMostIdx--) {
                currLevel = vLocalReconstructLevels[localArrayStartIdx + leftMostIdx];
                if (currLevel != borderLevel)
                    break;
            }
            leftMostIdx++;
            assert(vLocalReconstructLevels[localArrayStartIdx + leftMostIdx] == borderLevel);
        }

        // part 2. do shared-node driven permutation
        rightBoundIdx = length - 2;
        assert(leftMostIdx <= rightBoundIdx);
        if (leftMostIdx < rightBoundIdx) {
            int i, lit1, lit2, lit3;
            lit1 = vLocalReconstructArrays[localArrayStartIdx + rightBoundIdx + 1];
            lit2 = vLocalReconstructArrays[localArrayStartIdx + rightBoundIdx];

            for (i = rightBoundIdx; i >= leftMostIdx; i--) {
                lit3 = vLocalReconstructArrays[localArrayStartIdx + i];
                if (
                    retrieveAndNode(lit1, lit3, htReconstructKeys, htReconstructValues, htReconstructCapacity) 
                    != HASHTABLE_EMPTY_VALUE<uint64, uint32>
                ) {
                    if (lit3 != lit2) {
                        int tempLevel;
                        vLocalReconstructArrays[localArrayStartIdx + i] = lit2;
                        vLocalReconstructArrays[localArrayStartIdx + rightBoundIdx] = lit3;

                        tempLevel = vLocalReconstructLevels[localArrayStartIdx + i];
                        vLocalReconstructLevels[localArrayStartIdx + i] = vLocalReconstructLevels[localArrayStartIdx + rightBoundIdx];
                        vLocalReconstructLevels[localArrayStartIdx + rightBoundIdx] = tempLevel;
                    }
                    break;
                }
            }
        }
    }
}

void getTopTwoNodes(int *vLocalReconstructArrays, int *vLocalReconstructLevels,
                    int *vLocalReconstructLens, const uint64 *htReconstructKeys,
                    const uint32 *htReconstructValues,
                    const int htReconstructCapacity, const uint32 *htLevelKeys,
                    const uint32 *htLevelValues, const int htLevelCapacity,
                    uint64 *vStepNodeKeys, uint32 *vStepNodeValues,
                    int *vStepNodeMask, uint32 *vStepNodeLevels,
                    const int newIdCounter, const int nReadyCovers,
                    const int nPIs, const int nPOs, const int maxCoverLen,
                    int *dMaxMask) {
    sycl::queue &q_ct1 = nsycl::queue();
    // permute local reconstruct arrays, get top two for each cover, 
    // prepare them in hashtable key format and form 
    // 1) consecutive arrays of keys and values; 2) a mask indicating whether is null or trivial cases; 3) new node levels.
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nReadyCovers, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            sharedNodeDrivenPermute(
                vLocalReconstructArrays, vLocalReconstructLevels,
                vLocalReconstructLens, htReconstructKeys, htReconstructValues,
                htReconstructCapacity, nReadyCovers, maxCoverLen);
        });
    // (removed redundant device sync: in-order queue preserves ordering)
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nReadyCovers, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            prepareDataToInsert(
                vLocalReconstructArrays, vLocalReconstructLevels,
                vLocalReconstructLens, vStepNodeKeys, vStepNodeValues,
                vStepNodeMask, vStepNodeLevels, htReconstructKeys,
                htReconstructValues, htReconstructCapacity, htLevelKeys,
                htLevelValues, htLevelCapacity, newIdCounter, nReadyCovers,
                nPIs, nPOs, maxCoverLen, dMaxMask);
        });
    // (removed redundant device sync: the caller's oneDPL reduce over
    //  vStepNodeMask on the same in-order queue orders after this kernel)
}

void prepareReconstructArrays(const int * vGatheredReadyCovers, const int * vNodeCoverIdMapping, 
                                         const int * vCoverNodeIdMapping, const int * vCoverNodeNewLitMapping, 
                                         const int * vCoverTable, const int * vCoverTableLinks, const int * vCoverTableLens, 
                                         const uint32 * htLevelKeys, const uint32 * htLevelValues, const int htLevelCapacity,
                                         int * vLocalReconstructArrays, int * vLocalReconstructLevels, int * vLocalReconstructLens, 
                                         const int nReadyCovers, const int nPIs, const int maxCoverLen,
                                         const int sortDecId) {
    // gather cover input list from global cover table to local reconstruct arrays,
    // then do sorting
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nReadyCovers) {
        // int coverId = vGatheredReadyCovers[idx];
        // int oldOutputNodeId = vCoverNodeIdMapping[coverId];
        int oldOutputNodeId = vGatheredReadyCovers[idx];
        int currRowIdx = oldOutputNodeId;
        int columnPtr = 0;
        int length = vCoverTableLens[oldOutputNodeId];
        int currInputLit, currInputOldId, currInputNewLit, currInputNewId;
        int localArrayStartIdx = idx * maxCoverLen;
        assert(length <= maxCoverLen);

        // if (oldOutputNodeId == 480634 || oldOutputNodeId == 480396)
        //     printf("prepare reconstruct %d, len = %d\n", oldOutputNodeId, length);

        vLocalReconstructLens[idx] = length;
        if (length == 0) return;

        // int vRand[120];
        // assert(length <= 120);
        oneapi::dpl::default_engine rng(idx);
        // thrust::uniform_int_distribution<int> dist(0, 9999999);


        for (int i = 0; i < length; i++) {
            // vRand[i] = dist(rng);

            // change row
            if (columnPtr == COVER_TABLE_NUM_COLS) {
                columnPtr = 0;
                currRowIdx = vCoverTableLinks[currRowIdx];
            }
            currInputLit = vCoverTable[currRowIdx * COVER_TABLE_NUM_COLS + (columnPtr++)];
            currInputOldId = dUtils::AigNodeID(currInputLit);
            // if (oldOutputNodeId == 480634 || oldOutputNodeId == 480396)
            //     printf("%s%d -> ", dUtils::AigNodeIsComplement(currInputLit) ? "!" : "", currInputOldId);

            // // new id of PIs are same as old id, and also not saved in new id mapping
            // if (isPI)
            //     currInputNewId = currInputOldId;
            // else
            //     currInputNewId = vCoverNodeNewIdMapping[vNodeCoverIdMapping[currInputOldId]];
            // assert(currInputNewId != -1);
            
            // // save new literal into local reconstruct array
            // vLocalReconstructArrays[localArrayStartIdx + i] = dUtils::AigNodeLitCond(
            //     currInputNewId, dUtils::AigNodeIsComplement(currInputLit)
            // );

            // // save level info
            // if (isPI) {
            //     // zero level for PIs
            //     vLocalReconstructLevels[localArrayStartIdx + i] = 0;
            // } else {
            //     // retrieve level info from level hash table
            //     int retrieveRes;
            //     retrieveRes = retrieve_single<uint32, uint32>(
            //         htLevelKeys, htLevelValues, (uint32)currInputNewId, htLevelCapacity
            //     );
            //     assert(retrieveRes != (HASHTABLE_EMPTY_VALUE<uint32, uint32>));
            //     vLocalReconstructLevels[localArrayStartIdx + i] = retrieveRes;
            // }

            if (dUtils::AigIsPIConst(currInputOldId, nPIs)) {
                // save new literal into local reconstruct array
                vLocalReconstructArrays[localArrayStartIdx + i] = currInputLit;
                // zero level for PIs
                vLocalReconstructLevels[localArrayStartIdx + i] = 0;
            } else {
                currInputNewLit = vCoverNodeNewLitMapping[vNodeCoverIdMapping[currInputOldId]];
                // save new literal into local reconstruct array
                vLocalReconstructArrays[localArrayStartIdx + i] = dUtils::AigNodeNotCond(
                    currInputNewLit, dUtils::AigNodeIsComplement(currInputLit)
                );
                currInputNewId = dUtils::AigNodeID(currInputNewLit);
                // if (oldOutputNodeId == 480634 || oldOutputNodeId == 480396)
                //     printf("%s%d, ", dUtils::AigNodeIsComplement(currInputLit) ? "!" : "", currInputNewId);

                // save level info
                if (dUtils::AigIsPIConst(currInputNewId, nPIs))
                    vLocalReconstructLevels[localArrayStartIdx + i] = 0;
                else {
                    // retrieve level info from level hash table
                    int retrieveRes;
                    retrieveRes = retrieve_single<uint32, uint32>(
                        htLevelKeys, htLevelValues, (uint32)currInputNewId, htLevelCapacity
                    );
                    assert(retrieveRes != (HASHTABLE_EMPTY_VALUE<uint32, uint32>));
                    vLocalReconstructLevels[localArrayStartIdx + i] = retrieveRes;
                }
                
            }
        }
        // if (oldOutputNodeId == 480634 || oldOutputNodeId == 480396)
        //     printf("\n");

        // sort in desecending level order, no further parallel
        // thrust::sort_by_key(
        //     thrust::seq, &vLocalReconstructLevels[localArrayStartIdx], 
        //     &vLocalReconstructLevels[localArrayStartIdx + length], 
        //     &vLocalReconstructArrays[localArrayStartIdx],
        //     thrust::greater<int>()
        // );

        // Device-side sort of the two parallel local arrays. oneDPL's seq sort
        // and thrust::shuffle are not usable in SYCL device code (they pull in
        // recursive std::sort / unsupported thrust), so we sort by hand. Any
        // valid ordering yields a functionally-equivalent balanced tree.
        {
            int *A = &vLocalReconstructArrays[localArrayStartIdx];
            int *L = &vLocalReconstructLevels[localArrayStartIdx];
            if (sortDecId) {
                // insertion sort: descending level, tie-break descending id
                for (int a = 1; a < length; a++) {
                    int av = A[a], lv = L[a];
                    int b = a - 1;
                    while (b >= 0 && (L[b] < lv || (L[b] == lv && A[b] < av))) {
                        A[b + 1] = A[b]; L[b + 1] = L[b]; b--;
                    }
                    A[b + 1] = av; L[b + 1] = lv;
                }
            } else {
                // Fisher-Yates shuffle (random tie-breaks) then stable
                // insertion sort by descending level only
                for (int a = length - 1; a > 0; a--) {
                    int r = (int)(rng() % (unsigned)(a + 1));
                    int t = A[a]; A[a] = A[r]; A[r] = t;
                    t = L[a]; L[a] = L[r]; L[r] = t;
                }
                for (int a = 1; a < length; a++) {
                    int av = A[a], lv = L[a];
                    int b = a - 1;
                    while (b >= 0 && L[b] < lv) {
                        A[b + 1] = A[b]; L[b + 1] = L[b]; b--;
                    }
                    A[b + 1] = av; L[b + 1] = lv;
                }
            }
        }
        

        
    }
}


void addBackLocalArrays(int * vLocalReconstructArrays, int * vLocalReconstructLevels, int * vLocalReconstructLens, 
                                   const uint32 * vStepNodeValues, const int * vStepNodeMask, const uint32 * vStepNodeLevels,
                                   const int nReadyCovers, const int maxCoverLen, uint32 * dMaxId) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    // mask = 0 means that idx has complete reconstruction in previous steps
    if (idx < nReadyCovers && vStepNodeMask[idx] != 0) {
        // maxId is the max value over entries with mask == 1 (newly created
        // nodes); fold the reduction in here to avoid a per-step oneDPL pass.
        if (vStepNodeMask[idx] == 1)
            nsycl::atomic_fetch_max(
                dMaxId, vStepNodeValues[idx]);
        int litToInsert = (vStepNodeMask[idx] == 2 ? vStepNodeValues[idx] : (vStepNodeValues[idx] << 1));
        int localArrayStartIdx = idx * maxCoverLen;
        int length = vLocalReconstructLens[idx];
        int i;
        int temp;

        // check whether there alreay exists a same node
        for (i = 0; i < length; i++) {
            if (vLocalReconstructArrays[localArrayStartIdx + i] == litToInsert)
                return;
        }

        // insert
        vLocalReconstructArrays[localArrayStartIdx + length] = litToInsert;
        vLocalReconstructLevels[localArrayStartIdx + length] = vStepNodeLevels[idx];
        vLocalReconstructLens[idx]++;

        // adjust new node to proper location in terms of level
        for (i = length; i > 0; i--) {
            if (vLocalReconstructLevels[localArrayStartIdx + i] <= 
                vLocalReconstructLevels[localArrayStartIdx + i - 1])
                break;

            temp = vLocalReconstructArrays[localArrayStartIdx + i];
            vLocalReconstructArrays[localArrayStartIdx + i] = vLocalReconstructArrays[localArrayStartIdx + i - 1];
            vLocalReconstructArrays[localArrayStartIdx + i - 1] = temp;

            temp = vLocalReconstructLevels[localArrayStartIdx + i];
            vLocalReconstructLevels[localArrayStartIdx + i] = vLocalReconstructLevels[localArrayStartIdx + i - 1];
            vLocalReconstructLevels[localArrayStartIdx + i - 1] = temp;
        }
    }
}

void recordReconstructedCovers(const int * vLocalReconstructArrays, const int * vLocalReconstructLens, 
                                          const int * vGatheredReadyCovers, const int * vNodeCoverIdMapping, int * vCoverNodeNewLitMapping, 
                                          const int nReadyCovers, const int maxCoverLen) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nReadyCovers) {
        int newLit;
        if (vLocalReconstructLens[idx] == 0) {
            // constant false
            newLit = dUtils::AigConst0;
            // if (vGatheredReadyCovers[idx] == 480634)
            //     printf("** constant false!\n");
        } else {
            assert(vLocalReconstructLens[idx] == 1);
            newLit = vLocalReconstructArrays[idx * maxCoverLen];
            // if (vGatheredReadyCovers[idx] == 480634)
            //     printf("** normal branch, newLit=%d!\n", newLit);
        }
        
        int nodeId = vGatheredReadyCovers[idx];
        int coverId = vNodeCoverIdMapping[nodeId];
        vCoverNodeNewLitMapping[coverId] = newLit;
    }
}

void genReadyMask(const int * vCoverNodeIdMapping, const int * vNodeCoverIdMapping, 
                             const int * vCoverTable, const int * vCoverTableLinks, const int * vCoverTableLens, 
                             const int * vCoverNodeNewLitMapping, int * vThisIterReady, const int arrayLen, const int nPIs) {
    // vCoverNodeNewLitMapping indicates ready or not
    // generate a mask vThisIterReady for all covers that can be re-constructed in this iter
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx > 0 && idx < arrayLen) {
        // if cover to new id mapping is not -1, meaning the cover is already re-constructed
        if (vCoverNodeNewLitMapping[idx] >= 0) {
            // already constructed covers, should not construct again
            vThisIterReady[idx] = 0;
        } else {
            // check ready status by going through all its inputs
            int nodeId = vCoverNodeIdMapping[idx];
            int ready = 1;
            int currRowIdx = nodeId;
            int columnPtr = 0;
            int length = vCoverTableLens[nodeId];
            int currInputId;

            for (int i = 0; i < length; i++) {
                // change row
                if (columnPtr == COVER_TABLE_NUM_COLS) {
                    columnPtr = 0;
                    currRowIdx = vCoverTableLinks[currRowIdx];
                }
                currInputId = dUtils::AigNodeID(
                    vCoverTable[currRowIdx * COVER_TABLE_NUM_COLS + (columnPtr++)]
                );

                if (dUtils::AigIsPIConst(currInputId, nPIs))
                    continue;

                if (vCoverNodeNewLitMapping[vNodeCoverIdMapping[currInputId]] == -1) {
                    // this input is not re-constructed
                    ready = 0;
                    break;
                }
            }

            vThisIterReady[idx] = ready;
        }
    }
}

void gatherByScannedMask(int * vScannedMask, int * vGathered, const int arrayLen, const int idOffset=0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < arrayLen) {
        if (idx == 0) {
            // vGathered[vScannedMask[idx] - 1] = idx + idOffset
            if (vScannedMask[0] == 1)
                vGathered[0] = idOffset;
        } else {
            // + idOffset indicates id start from idOffset
            if (vScannedMask[idx] > vScannedMask[idx - 1])
                vGathered[vScannedMask[idx] - 1] = idx + idOffset;
        }
    }
}

int gatherByMask(int *vMask, int *vGathered, const int arrayLen,
                 const int idOffset) {
    sycl::queue &q_ct1 = nsycl::queue();
    // gathered results will be saved in vGathered, make sure vGathered has at least arrayLen entries
    // contents in vMask will be modified
    int nGathered = -1;

    oneapi::dpl::inclusive_scan(
        oneapi::dpl::execution::make_device_policy(q_ct1), vMask,
        vMask + arrayLen, vMask);
    q_ct1.wait();
    q_ct1.memcpy(&nGathered, &vMask[arrayLen - 1], sizeof(int)).wait();
    assert(nGathered != -1);

    if (nGathered > 0) {
        q_ct1.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(arrayLen, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                gatherByScannedMask(vMask, vGathered, arrayLen, idOffset);
            });
    }
    return nGathered;
}

int getReadyCovers(const int * vCoverNodeIdMapping, const int * vNodeCoverIdMapping, 
                   const int * vCoverTable, const int * vCoverTableLinks, const int * vCoverTableLens, 
                   const int * vCoverNodeNewLitMapping, int * vThisIterReady, int * vGatheredReadyCovers, 
                   const int nCovers, const int nPIs) {
    // get all ready cover ids, and the total number of ready covers
    int nReadyCovers = -1;

    nsycl::queue().submit([&](sycl::handler &cgh) {
        auto nCovers_ct7 = nCovers + 1;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1,
                               NUM_BLOCKS(nCovers + 1, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                genReadyMask(vCoverNodeIdMapping, vNodeCoverIdMapping,
                             vCoverTable, vCoverTableLinks, vCoverTableLens,
                             vCoverNodeNewLitMapping, vThisIterReady,
                             nCovers_ct7, nPIs);
            });
    });

    nReadyCovers = gatherByMask(vThisIterReady + 1, vGatheredReadyCovers, nCovers, 1);

    return nReadyCovers;
}


void markIsCoverOutput(const int * vCoverTableLens, int * vMarks, const int arrayLen) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < arrayLen) {
        vMarks[idx] = (vCoverTableLens[idx] != -1 ? 1 : 0);
    }
}

void getCoverToNodeIdMapping(const int * vNodeCoverIdMapping, const int * vCoverTableLens, 
                                        int * vCoverNodeIdMapping, const int arrayLen) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < arrayLen) {
        if (vCoverTableLens[idx] != -1) {
            // idx is output node id that corresponds to a cover
            vCoverNodeIdMapping[vNodeCoverIdMapping[idx]] = idx;
        }
    }
}

void gatherWithFilter(int * isEnd, 
                                 const int * vCoverTable, const int * vCoverTableLinks, 
                                 const int * vNodes, const int * vCoverRanges, int * vCanonTable, 
                                 int * vNewGlobalList, int * vNodesFilter, const int nPIs, const int arrayLen) {
    // vCanonTable should be all zero before this kernel call
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < arrayLen) {
        int dstStartIdx = (idx == 0 ? 0 : vCoverRanges[idx - 1]);
        // int length =  vCoverRanges[idx] - dstStartIdx;
        int dstEndIdx = vCoverRanges[idx];

        int fatherLit = vNodes[idx];
        int fatherId = dUtils::AigNodeID(fatherLit);
        // if (length != vCoverTableLens[fatherId]) {
        //     printf("father id %d: range %d, table len %d\n", fatherId, length, vCoverTableLens[fatherId]);
        // }
        // assert(length == vCoverTableLens[fatherId]);

        int currRowIdx, columnPtr;
        currRowIdx = fatherId;
        columnPtr = 0;

        int laterThread, currLit, currId;
        for (int i = dstStartIdx; i < dstEndIdx; i++) {
            // change row
            if (columnPtr == COVER_TABLE_NUM_COLS) {
                columnPtr = 0;
                currRowIdx = vCoverTableLinks[currRowIdx];
            }
            currLit = vCoverTable[currRowIdx * COVER_TABLE_NUM_COLS + columnPtr];
            columnPtr++;
            currId = dUtils::AigNodeID(currLit);
            vNewGlobalList[i] = currLit;

            // check whether this is PI
            if (!dUtils::AigIsNode(currId, nPIs)) {
                // mark in filter
                vNodesFilter[i] = 0;
                continue;
            }

            // update father, only one entry corresponds to currId will succeed
            laterThread = nsycl::atomic_compare_exchange(
                &vCanonTable[currId], 0, 1);
            // laterThread = atomicAdd(&vCanonTable[currId], 1);
            if (!laterThread) {
                // first thread, perform update
                // mark as non-PI, non-R
                vNodesFilter[i] = 1;
                // *isEnd = 0;
            } else {
                // later thread, mark itself as duplicate in filter
                vNodesFilter[i] = -1;
            }
        }
    }
}

int localCoverTravToTable(int nodeLit, const int *pFanin0, const int *pFanin1,
                          const int *pNumFanouts, int *vCoverTable,
                          int *vCoverTableLinks, int *vCoverTableLens,
                          int *nCoverTableNext, int *superLen, const int nPIs) {
    // because of the condition "pNumFanouts[nodeId] > 1", the traversal only visits a tree-structure, 
    // therefore even there is no visited marks, we will never encounter repeated nodes.
    // also saves traversal results into the table
    int nodeId, outputNodeId;
    int i, currLit, checkLit, samePolarFlag;
    int currRowIdx, lastRowIdx, columnPtr;
    int checkCurrRowIdx, checkColumnPtr; // used in checking existing nodes

    int stack[DFS_COVER_STACK_SIZE];
    int stackTop = -1;
    stack[++stackTop] = nodeLit;

    // initial changes on cover table
    outputNodeId = dUtils::AigNodeID(nodeLit);
    vCoverTableLinks[outputNodeId] = 0;
    // writing pointers
    currRowIdx = outputNodeId;
    columnPtr = 0;

    while (stackTop != -1) {
        // pop
        currLit = stack[stackTop--];

        // check whether currLit is already in cover table
        samePolarFlag = 0;
        checkCurrRowIdx = outputNodeId;
        for (i = 0, checkColumnPtr = 0; i < *superLen; i++, checkColumnPtr++) {
            // change row
            if (checkColumnPtr == COVER_TABLE_NUM_COLS) {
                checkColumnPtr = 0;
                assert(vCoverTableLinks[checkCurrRowIdx] > checkCurrRowIdx);
                checkCurrRowIdx = vCoverTableLinks[checkCurrRowIdx];
            }
            checkLit = vCoverTable[checkCurrRowIdx * COVER_TABLE_NUM_COLS + checkColumnPtr];

            if (checkLit == currLit) {
                // same polarity
                samePolarFlag = 1;
                break;
            } else if (checkLit == dUtils::AigNodeNot(currLit)) {
                // opposite polarity, modify cover table and directly return false
                vCoverTableLens[outputNodeId] = 0;
                return 0;
            }
        }

        // if currLit has a replica in cover table, then prune this branch
        if (samePolarFlag)
            continue;
        
        // encounter cover input nodes
        nodeId = dUtils::AigNodeID(currLit);
        // if (currLit != nodeLit && (dUtils::AigNodeIsComplement(currLit) || !dUtils::AigIsNode(nodeId, nPIs) || pNumFanouts[nodeId] > 1)) {
        if (currLit != nodeLit && 
            (dUtils::AigNodeIsComplement(currLit) || !dUtils::AigIsNode(nodeId, nPIs) || pNumFanouts[nodeId] > 1 
                || stackTop+2>=DFS_COVER_STACK_SIZE || *superLen+2*(stackTop+1)+1 >= MAX_LOCAL_COVER_SIZE )
            ) {
            // add one node into cover table
            if (columnPtr == COVER_TABLE_NUM_COLS) {
                // expand a new row
                lastRowIdx = currRowIdx;
                currRowIdx = nsycl::atomic_fetch_add(nCoverTableNext,
                                                                1);
                vCoverTableLinks[lastRowIdx] = currRowIdx;
                vCoverTableLinks[currRowIdx] = 0;
                columnPtr = 0;
            }
            vCoverTable[currRowIdx * COVER_TABLE_NUM_COLS + columnPtr] = currLit;
            columnPtr++;

            (*superLen)++;
            continue;
        }

        // in reversed order to get identical result as the recursive algorithm
        stack[++stackTop] = pFanin1[nodeId];
        stack[++stackTop] = pFanin0[nodeId];

        assert(stackTop < DFS_COVER_STACK_SIZE);
    }

    vCoverTableLens[outputNodeId] = *superLen;
    return 1;
}

void coverFindingToTable(int * vNodes, int * vNodesStatus, int * vLastAppearLevel, 
                                    const int * pFanin0, const int * pFanin1, const int * pNumFanouts, 
                                    int * vCoverTable, int * vCoverTableLinks, int * vCoverTableLens, int * nCoverTableNext,
                                    const int nPIs, const int arrayLen, const int levelCount) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < arrayLen) {
        // only perform traversal for vNodesFilter[idx] = 1
        // if (vNodesFilter[idx] != 1) {
        //     vCoverWiseLens[idx] = 0;
        //     return;
        // }

        // int thisLit = vNodes[idx];
        // int nodeId = dUtils::AigNodeID(thisLit); // note, we assume that there is no concurrent thread with same nodeId
        int nodeId = vNodes[idx];
        int thisLit = (nodeId << 1); // the polarity of nodeId will not affect cover traversal
        int superLen = 0;
        int ret;

        // update last appear level of this cover output node
        vLastAppearLevel[nodeId] = levelCount;

        if (vCoverTableLinks[nodeId] != -1) {
            // assuming that there are no duplicates in vNodes, this means that
            // this cover has already been discovered in the previous levels;
            // if there are duplicates, then this part should be modified
            superLen = vCoverTableLens[nodeId];
            // if superLen is zero, then this cover should be constant true/false
            ret = (superLen != 0);
        } else {
            // do cover traversal and write into cover table
            ret = localCoverTravToTable(
                thisLit, pFanin0, pFanin1, pNumFanouts, 
                vCoverTable, vCoverTableLinks, vCoverTableLens, nCoverTableNext,
                &superLen, nPIs
            );
        }

        if (!ret) {
            // has two opposite edges
            // note that for constant false covers, the status array is not updated

            // printf("constant false triggered! nodeId: %d\n", nodeId);
            // vCoverWiseLens[idx] = 0;
            // vNodesFilter[idx] = 2; // 2 indicates constant true/false
            // if (dUtils::AigNodeIsComplement(thisLit)) // constant true
            //     vNodes[idx] = dUtils::AigConst1;
            // else                                      // constant false
            //     vNodes[idx] = dUtils::AigConst0;
            vNodes[idx] = 0;
        } else {
            assert(superLen > 1);
            assert(superLen <= MAX_LOCAL_COVER_SIZE);
            // vCoverWiseLens[idx] = superLen;
            
            // assign corresponding entries in the status array to 1
            int currRowIdx, columnPtr;
            currRowIdx = nodeId;
            columnPtr = 0;

            int currLit, currId;
            for (int i = 0; i < superLen; i++) {
                // change row
                if (columnPtr == COVER_TABLE_NUM_COLS) {
                    columnPtr = 0;
                    currRowIdx = vCoverTableLinks[currRowIdx];
                }
                currLit = vCoverTable[currRowIdx * COVER_TABLE_NUM_COLS + columnPtr];
                columnPtr++;
                currId = dUtils::AigNodeID(currLit);
                if (dUtils::AigIsNode(currId, nPIs))
                    vNodesStatus[currId] = 1;
                // if (nodeId == 480634) {
                //     printf("%d ", currId);
                // }
            }
            // if (nodeId == 480634)
            //     printf("\n");
        }

    }
}

void findLevelNodeRanges(const int * vLastAppearLevel, int * vLastAppearLevelRanges, const int nCovers,
                         const sycl::stream &stream_ct1) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx == nCovers - 1) {
        stream_ct1 << "*** Max Level: " << vLastAppearLevel[idx] << "\n";
        vLastAppearLevelRanges[vLastAppearLevel[idx]] = idx + 1;
    } else if (idx < nCovers - 1) {
        if (vLastAppearLevel[idx] < vLastAppearLevel[idx + 1]) {
            vLastAppearLevelRanges[vLastAppearLevel[idx]] = idx + 1;
        }
    }
}

void parseOutputRes(const uint64 * vReconstructedKeys, 
                               const uint32 * htOutKeys, const uint32 * htOutValues, const int htOutCapacity,
                               int * vFanin0New, int * vFanin1New, int * vNumFanoutsNew, 
                               const int nEntries, const int nPIs) {
    // NOTE vFanin0New, vFanin1New should point to the begin of AND node storage;
    //      vNumFanoutsNew should point to the begin of PIs + AND node storage!
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nEntries) {
        uint64 key = vReconstructedKeys[idx];
        uint32 oldLit0, oldLit1, oldId0, oldId1, newId0, newId1;
        unbindAndNodeKeys(key, &oldLit0, &oldLit1);

        oldId0 = dUtils::AigNodeID(oldLit0);
        oldId1 = dUtils::AigNodeID(oldLit1);

        if (dUtils::AigIsPIConst(oldId0, nPIs))
            newId0 = oldId0;
        else
            newId0 = retrieve_single<uint32, uint32>(htOutKeys, htOutValues, oldId0, htOutCapacity);
        
        if (dUtils::AigIsPIConst(oldId1, nPIs))
            newId1 = oldId1;
        else
            newId1 = retrieve_single<uint32, uint32>(htOutKeys, htOutValues, oldId1, htOutCapacity);
        assert(newId0 != (HASHTABLE_EMPTY_VALUE<uint32, uint32>));
        assert(newId1 != (HASHTABLE_EMPTY_VALUE<uint32, uint32>));

        vFanin0New[idx] = dUtils::AigNodeLitCond(newId0, dUtils::AigNodeIsComplement(oldLit0));
        vFanin1New[idx] = dUtils::AigNodeLitCond(newId1, dUtils::AigNodeIsComplement(oldLit1));
        nsycl::atomic_fetch_add(
            &vNumFanoutsNew[newId0], 1);
        nsycl::atomic_fetch_add(
            &vNumFanoutsNew[newId1], 1);
    }
}

void processPO(const int * d_pOuts, const int * vNodeCoverIdMapping, const int * vCoverNodeNewLitMapping, 
                          const uint32 * htOutKeys, const uint32 * htOutValues, const int htOutCapacity,
                          int * vNewOuts, int * vNumFanoutsNew, const int nPOs, const int nPIs,
                          const sycl::stream &stream_ct1) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx < nPOs) {
        int oldLit = d_pOuts[idx];
        int oldId = dUtils::AigNodeID(oldLit);

        // increase number of fanout for POs, which ensures consistency with ABC
        if (dUtils::AigIsPIConst(oldId, nPIs)) {
            vNewOuts[idx] = oldLit;
            nsycl::atomic_fetch_add(
                &vNumFanoutsNew[oldId], 1);
        } else {
            int newLit = vCoverNodeNewLitMapping[vNodeCoverIdMapping[oldId]];
            assert(newLit != -1);
            int outId = retrieve_single<uint32, uint32>(htOutKeys, htOutValues, dUtils::AigNodeID(newLit), htOutCapacity);
            if (outId == (HASHTABLE_EMPTY_VALUE<uint32, uint32>)) {
                // should be PI or const
                stream_ct1 << "oldId: %d, coverId: %d, newLit: %d\n";
                outId = dUtils::AigNodeID(newLit);
                assert(dUtils::AigIsPIConst(outId, nPIs));
            }
            assert(outId != (HASHTABLE_EMPTY_VALUE<uint32, uint32>));

            vNewOuts[idx] = dUtils::AigNodeLitCond(outId, dUtils::AigNodeIsComplement(oldLit)^dUtils::AigNodeIsComplement(newLit));
            nsycl::atomic_fetch_add(
                &vNumFanoutsNew[outId], 1);
        }
    }
}

std::tuple<int *, int *, int *, int *, int>
balancePerformV2(int nObjs, int nPIs, int nPOs, int nNodes, int *d_pFanin0,
                 int *d_pFanin1, int *d_pOuts, int *d_pNumFanouts,
                 int sortDecId) {
    sycl::queue &q_ct1 = nsycl::queue();
    int levelCount = 0;
    int * vNodes, * vNodes2; // global input lists, double buffer, exchanged per iteration
    int * vNodesStatus;
    int * vNodesIndices;
    int * vNodesFilter;      // global input list filter
    int * vCoverWiseLens;    // lengths for each cover
    int * vCoverRanges;      // range for gathering local inputs of each cover
    
    int * vCanonTable;       // for duplicate checking
    int * vLastAppearLevel;  // last appear level for each cover
    int * vLastAppearLevelNodes;  // used together with vLastAppearLevel, for saving node id 
    int * vLastAppearLevelRanges; // indices of ranges for each level

    // cover table
    int * vCoverTable;
    int * vCoverTableLinks;  // indicating idx of next row, if one row in vCoverTable is not enough. -1: unvisited, 0: last row, >0: next row idx
    int * vCoverTableLens;
    int * nCoverTableNext;

    // levelized data structures

    // ******* Phase 1. cover finding without levelized recording *******
    
    auto start_t = hrClock();
    
    int globalListLen = nPOs > 131072 ? nPOs : 131072; // for dynamically increasing global list length
    int globalListLen2 = globalListLen;                // dynamically increase double buffer and vNodesFilter length

    vNodes = sycl::malloc_device<int>(nObjs, q_ct1);
    vNodes2 = sycl::malloc_device<int>(globalListLen2, q_ct1);
    vNodesFilter = sycl::malloc_device<int>(globalListLen2, q_ct1);
    vCoverWiseLens = sycl::malloc_device<int>(globalListLen, q_ct1);
    vCoverRanges = sycl::malloc_device<int>(globalListLen, q_ct1);
    q_ct1.memcpy(vNodes, d_pOuts, nPOs * sizeof(int));
    vNodesStatus = sycl::malloc_device<int>(nObjs, q_ct1);
    vNodesIndices = sycl::malloc_device<int>(nObjs, q_ct1);

    vCanonTable = sycl::malloc_device<int>(nObjs, q_ct1);
    vLastAppearLevel = sycl::malloc_device<int>(nObjs, q_ct1);
    vLastAppearLevelNodes = sycl::malloc_device<int>(nObjs, q_ct1);
    q_ct1.memset(vCanonTable, 0, nObjs * sizeof(int)).wait();
    q_ct1.memset(vLastAppearLevel, -1, nObjs * sizeof(int)).wait();

    vCoverTable =
        sycl::malloc_device<int>(2 * nObjs * COVER_TABLE_NUM_COLS, q_ct1);
    vCoverTableLinks = sycl::malloc_device<int>(2 * nObjs, q_ct1);
    vCoverTableLens = sycl::malloc_device<int>(nObjs, q_ct1);
    nCoverTableNext = sycl::malloc_device<int>(1, q_ct1);
    q_ct1.memset(vCoverTableLinks, -1, 2 * nObjs * sizeof(int)).wait();
    q_ct1.memset(vCoverTableLens, -1, nObjs * sizeof(int)).wait();
    q_ct1.memcpy(nCoverTableNext, &nObjs, sizeof(int)).wait();

    isEnd[0] = 1;
    q_ct1.wait();

    // len of current level global list
    int currLen = nPOs, newLen;
    // filter and convert elements in vNodes to node ids
    q_ct1.submit([&](sycl::handler &cgh) {
        auto isEnd_ct0 = isEnd.get_ptr();

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(nPOs, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                checkCoverTravEnd(isEnd_ct0, vNodes, vNodesFilter, vCanonTable,
                                  nPIs, nPOs);
            });
    });
    q_ct1.wait();
    if (!isEnd[0]) {
        // remove duplicates
        int *pFilteredEnd = nsycl::remove_if(q_ct1, vNodes,
            vNodes + nPOs, vNodesFilter, dUtils::isNotOne<int>());
        currLen = pFilteredEnd - vNodes;
    }
    if (isEnd[0] || currLen == 0)
        return {NULL, NULL, NULL, NULL, 0};

    #ifdef DEBUG
        printLevelInfo<<<1, 1>>>(vNodes, vNodesFilter, 0, nPOs, nPIs, nPOs);
        cudaDeviceSynchronize();
    #endif

    // precompute a consecutive indices array for gathering uses
    nsycl::iota(q_ct1, vNodesIndices,
               vNodesIndices + nObjs);

    do {
        q_ct1.memset(vNodesStatus, 0, nObjs * sizeof(int)).wait();

        // cover findings, record into cover table and vCoverWiseLens
        q_ct1.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(currLen, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                coverFindingToTable(vNodes, vNodesStatus, vLastAppearLevel,
                                    d_pFanin0, d_pFanin1, d_pNumFanouts,
                                    vCoverTable, vCoverTableLinks,
                                    vCoverTableLens, nCoverTableNext, nPIs,
                                    currLen, levelCount);
            });
        q_ct1.wait();

        // gather the indices where vNodesStatus = 1 to the next level's vNodes
        int *pNewGlobalListEnd = nsycl::copy_if(q_ct1, vNodesIndices,
            vNodesIndices + nObjs, vNodesStatus, vNodes, dUtils::isOne<int>());
        newLen = pNewGlobalListEnd - vNodes;

        // update currLen
        currLen = newLen;

        levelCount++;
        #ifdef DEBUG
            printf("Level %d, global list len %d\n", levelCount, currLen);
            printLevelInfo<<<1, 1>>>(vNodes, vNodesFilter, levelCount, currLen, nPIs, nPOs);
            printLevelFilterStats(vNodesFilter, levelCount, currLen);
            cudaDeviceSynchronize();
        #endif

    } while (currLen > 0);

    q_ct1.wait();
    auto phase1_t = hrClock();

    printf("Phase 1 time: %lf\n", (phase1_t - start_t) / (double) NS_PER_SEC);

    // get maximum local input length
    int *pMaxCoverLen =
        std::max_element(oneapi::dpl::execution::make_device_policy(q_ct1),
                         vCoverTableLens, vCoverTableLens + nObjs);
    int maxCoverLen;
    q_ct1.wait();
    q_ct1.memcpy(&maxCoverLen, pMaxCoverLen, sizeof(int)).wait();
    if (1) {
        printf("Max cover len: %d\n", maxCoverLen);
    }
    

    // ******* Phase 2. cover reconstruction *******

    // dense cover-related data structures
    int nCovers;
    int * vNodeCoverIdMapping; // map cover output node id (old) to cover id
    int * vCoverNodeIdMapping; // map cover id to cover output node id (old)
    int * vCoverNodeNewLitMapping; // map cover id to new output node literal
    int * vThisIterReady; // use as temp array in gathering ready cover ids
    int * vGatheredReadyCovers; // storage of ready cover ids

    int * vLocalReconstructArrays; // 2d scratch memory for recording per-cover input ids during re-construction
    int * vLocalReconstructLevels; // 2d scratch memory for recording levels during re-construction
    int * vLocalReconstructLens;   // 1d scratch memory for recording per-cover input lens

    uint64 * vStepNodeKeys;        // memory for per-batch hash table insert and retrieval
    uint32 * vStepNodeValues;
    int * vStepNodeMask;
    uint32 * vStepNodeLevels;

    auto phase2_start_t = hrClock();

    vNodeCoverIdMapping = sycl::malloc_device<int>(nObjs, q_ct1);

    // hash tables
    HashTable<uint32, uint32> levelHashTable((int)(nObjs / HT_LOAD_FACTOR));
    HashTable<uint64, uint32> reconstructHashTable((int)(nObjs / HT_LOAD_FACTOR));
    // expose hash table storage
    uint32 * htLevelKeys = levelHashTable.get_keys_storage();
    uint32 * htLevelValues = levelHashTable.get_values_storage();
    int htLevelCapacity = levelHashTable.get_capacity();
    uint64 * htReconstructKeys = reconstructHashTable.get_keys_storage();
    uint32 * htReconstructValues = reconstructHashTable.get_values_storage();
    int htReconstructCapacity = reconstructHashTable.get_capacity();

    // 2.1 compute output node id to cover id mapping, and inverse mapping
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nObjs, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            markIsCoverOutput(vCoverTableLens, vNodeCoverIdMapping, nObjs);
        });
    // cudaDeviceSynchronize();
    // note that id ranges from 1 to n, so need to allocate n + 1 entries for related data structures
    oneapi::dpl::inclusive_scan(
        oneapi::dpl::execution::make_device_policy(q_ct1), vNodeCoverIdMapping,
        vNodeCoverIdMapping + nObjs, vNodeCoverIdMapping);
    q_ct1.wait();

    // get number of covers
    q_ct1.memcpy(&nCovers, &vNodeCoverIdMapping[nObjs - 1], sizeof(int)).wait();
    q_ct1.wait();

    printf("Total number of covers: %d\n", nCovers);
    
    // 2.1.1 compute cover ready status of each level
    vLastAppearLevelRanges = sycl::malloc_device<int>(levelCount, q_ct1);
    // assign initial node ids
    nsycl::iota(q_ct1,
               vLastAppearLevelNodes, vLastAppearLevelNodes + nObjs);
    int *pNewLastAppearLevelNodesEnd =
        nsycl::remove_if(q_ct1,
                        vLastAppearLevelNodes, vLastAppearLevelNodes + nObjs,
                        vLastAppearLevel, dUtils::isMinusOne<int>());
    int *pNewLastAppearLevelEnd =
        oneapi::dpl::remove(oneapi::dpl::execution::make_device_policy(q_ct1),
                            vLastAppearLevel, vLastAppearLevel + nObjs, -1);
    assert(pNewLastAppearLevelNodesEnd - vLastAppearLevelNodes == nCovers);
    assert(pNewLastAppearLevelEnd - vLastAppearLevel == nCovers);

    // sort in level no.
    oneapi::dpl::sort_by_key(oneapi::dpl::execution::make_device_policy(q_ct1),
               vLastAppearLevel, vLastAppearLevel + nCovers,
               vLastAppearLevelNodes);
    // find range indices for each level
    q_ct1.submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(nCovers, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                findLevelNodeRanges(vLastAppearLevel, vLastAppearLevelRanges,
                                    nCovers, stream_ct1);
            });
    });
    q_ct1.wait();

    // int iterLen = 100000000;
    const int iterLen = 32768;

    int * levelRanges = (int *) malloc(levelCount * sizeof(int));
    q_ct1.memcpy(levelRanges, vLastAppearLevelRanges, levelCount * sizeof(int))
        .wait();

    int maxReadyCovers = levelRanges[0];
    for(int i=1; i<levelCount; i++)
        maxReadyCovers =
            std::max(maxReadyCovers, levelRanges[i] - levelRanges[i - 1]);
    maxReadyCovers = std::min(maxReadyCovers, iterLen);

    vCoverNodeIdMapping = sycl::malloc_device<int>((nCovers + 1), q_ct1);
    vCoverNodeNewLitMapping = sycl::malloc_device<int>((nCovers + 1), q_ct1);
    vThisIterReady = sycl::malloc_device<int>((nCovers + 1), q_ct1);
    vGatheredReadyCovers = sycl::malloc_device<int>(nCovers, q_ct1);
    
        vLocalReconstructArrays = sycl::malloc_device<int>(
            (size_t)maxReadyCovers * (size_t)maxCoverLen, q_ct1);
    
        vLocalReconstructLevels = sycl::malloc_device<int>(
            (size_t)maxReadyCovers * (size_t)maxCoverLen, q_ct1);
    vLocalReconstructLens = sycl::malloc_device<int>(nCovers, q_ct1);
    vStepNodeKeys = sycl::malloc_device<uint64>(nCovers, q_ct1);
    vStepNodeValues = sycl::malloc_device<uint32>(nCovers, q_ct1);
    vStepNodeMask = sycl::malloc_device<int>(nCovers, q_ct1);
    vStepNodeLevels = sycl::malloc_device<uint32>(nCovers, q_ct1);

    q_ct1.memset(vCoverNodeNewLitMapping, -1, (nCovers + 1) * sizeof(int))
        .wait();
    q_ct1.wait();

    // compute inverse mapping
    q_ct1.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, NUM_BLOCKS(nObjs, THREAD_PER_BLOCK)) *
                sycl::range<3>(1, 1, THREAD_PER_BLOCK),
            sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            getCoverToNodeIdMapping(vNodeCoverIdMapping, vCoverTableLens,
                                    vCoverNodeIdMapping, nObjs);
        });
    q_ct1.wait();

    // 2.2 outer loop: check ready covers

    uint64 * vReconstructedKeys;
    uint32 * vReconstructedIds, * vOutIds;
    int * vNewOuts;
    vReconstructedKeys = sycl::malloc_device<uint64>(nObjs, q_ct1);
    vReconstructedIds = sycl::malloc_device<uint32>(nObjs, q_ct1);
    vOutIds = sycl::malloc_device<uint32>(nObjs, q_ct1);
    vNewOuts = sycl::malloc_device<int>(nPOs, q_ct1);

    int nReadyCovers, nReconstructed = 0;
    int newIdCounter = nPIs + 1;
    // small device scratch for per-step reductions folded into the kernels:
    // dReduce[0] = maxMask, dReduce[1] = maxId. Avoids two oneDPL device-policy
    // reductions (each of which allocates/frees USM temporaries) per inner step.
    int *dReduce = sycl::malloc_device<int>(2, q_ct1);
    int hReduce[2];

    // nReadyCovers = getReadyCovers(
    //     vCoverNodeIdMapping, vNodeCoverIdMapping, vCoverTable, vCoverTableLinks, vCoverTableLens, 
    //     vCoverNodeNewLitMapping, vThisIterReady, vGatheredReadyCovers, nCovers, nPIs
    // );
    levelCount--;

    while (levelCount >= 0) {
        int readyStartIdx = (levelCount == 0 ? 0 : levelRanges[levelCount - 1]);
        nReadyCovers = levelRanges[levelCount] - readyStartIdx;

        #ifdef DEBUG
            printf("Number of ready covers: %d\n", nReadyCovers);
        #endif
        nReconstructed += nReadyCovers;

        int innerExpandTimes = (nReadyCovers + iterLen - 1) / iterLen;
        for (int i = 0; i < innerExpandTimes; i++) {
            int thisReadyStartIdx = readyStartIdx + i * iterLen;
            int thisReadyCovers = (i == innerExpandTimes - 1 ? (nReadyCovers - (innerExpandTimes - 1) * iterLen) : iterLen);
            assert(thisReadyCovers > 0);

            // if OOM, the two local arrays can be dynamically increased with nReadyCovers
            q_ct1.submit([&](sycl::handler &cgh) {
                const int *vLastAppearLevelNodes_thisReadyStartIdx_ct0 =
                    vLastAppearLevelNodes + thisReadyStartIdx;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(
                            1, 1,
                            NUM_BLOCKS(thisReadyCovers, THREAD_PER_BLOCK)) *
                            sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                        sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
                    [=](sycl::nd_item<3> item_ct1) {
                        prepareReconstructArrays(
                            vLastAppearLevelNodes_thisReadyStartIdx_ct0,
                            vNodeCoverIdMapping, vCoverNodeIdMapping,
                            vCoverNodeNewLitMapping, vCoverTable,
                            vCoverTableLinks, vCoverTableLens, htLevelKeys,
                            htLevelValues, htLevelCapacity,
                            vLocalReconstructArrays, vLocalReconstructLevels,
                            vLocalReconstructLens, thisReadyCovers, nPIs,
                            maxCoverLen, sortDecId);
                    });
            });
            // cudaDeviceSynchronize();

            #ifdef DEBUG
                printLocalReconstructArray<<<1, 1>>>(
                    vLocalReconstructArrays, vLocalReconstructLens, vLocalReconstructLevels, 
                    thisReadyCovers, maxCoverLen, nPIs, nPOs
                );
                cudaDeviceSynchronize();
            #endif

            // 2.3 inner loop: perform one step of AND construction
            while (true) {
                // reset per-step reduction scratch (dReduce[0]=maxMask,
                // dReduce[1]=maxId); no host wait, in-order queue serialises it
                q_ct1.memset(dReduce, 0, 2 * sizeof(int));
                // gather batch data into vStep* arrays (also atomically
                // accumulates maxMask into dReduce[0])
                assert(INT_MAX - newIdCounter > thisReadyCovers);
                getTopTwoNodes(
                    vLocalReconstructArrays, vLocalReconstructLevels, vLocalReconstructLens, 
                    htReconstructKeys, htReconstructValues, htReconstructCapacity, 
                    htLevelKeys, htLevelValues, htLevelCapacity,
                    vStepNodeKeys, vStepNodeValues, vStepNodeMask, vStepNodeLevels, 
                    newIdCounter, thisReadyCovers, nPIs, nPOs, maxCoverLen,
                    dReduce
                );
                // check whether complete or not (single small copy-back)
                q_ct1.memcpy(&hReduce[0], &dReduce[0], sizeof(int)).wait();
                int maxMask = hReduce[0];
                assert(maxMask >= 0 && maxMask <= 2);
                if (maxMask == 0)
                    break;
                
                // TODO (future) actually can safely remove duplicate keys before insertion. since the new ids are generated by idx, 
                //      can traceback using this value. (deal with non-deterministic)
                
                // batch insert to hashtable
                reconstructHashTable.insert_batch_no_update_masked(vStepNodeKeys, vStepNodeValues, vStepNodeMask, thisReadyCovers);
                // retrieve immediately to update values in case there are duplicate keys in this batch
                reconstructHashTable.retrieve_batch_masked(vStepNodeKeys, vStepNodeValues, vStepNodeMask, thisReadyCovers);

                // for new nodes, insert the associated levels into level ht. note that if remove duplicate keys (previous TODO), 
                // this insertion can be done earlier in prepareDataToInsert.
                levelHashTable.insert_batch_no_update_masked(vStepNodeValues, vStepNodeLevels, vStepNodeMask, thisReadyCovers);

                // add result of this step back to local arrays while keeping the local array sorted w.r.t. level
                q_ct1.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(
                            1, 1,
                            NUM_BLOCKS(thisReadyCovers, THREAD_PER_BLOCK)) *
                            sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                        sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
                    [=](sycl::nd_item<3> item_ct1) {
                        addBackLocalArrays(
                            vLocalReconstructArrays, vLocalReconstructLevels,
                            vLocalReconstructLens, vStepNodeValues,
                            vStepNodeMask, vStepNodeLevels, thisReadyCovers,
                            maxCoverLen,
                            reinterpret_cast<uint32 *>(&dReduce[1]));
                    });
                // fetch maxId (folded into addBackLocalArrays above); used to
                // advance the id counter for the next step
                q_ct1.memcpy(&hReduce[1], &dReduce[1], sizeof(int)).wait();
                int maxId = hReduce[1];
                if (maxId != 0) {
                    // maxId = 0 means that all values are masked out
                    newIdCounter = maxId + 1;
                }
            }

            // 2.3 add newly constructed cover output nodes into vCoverNodeNewLitMapping
            q_ct1.submit([&](sycl::handler &cgh) {
                const int *vLastAppearLevelNodes_thisReadyStartIdx_ct2 =
                    vLastAppearLevelNodes + thisReadyStartIdx;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(
                            1, 1,
                            NUM_BLOCKS(thisReadyCovers, THREAD_PER_BLOCK)) *
                            sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                        sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
                    [=](sycl::nd_item<3> item_ct1) {
                        recordReconstructedCovers(
                            vLocalReconstructArrays, vLocalReconstructLens,
                            vLastAppearLevelNodes_thisReadyStartIdx_ct2,
                            vNodeCoverIdMapping, vCoverNodeNewLitMapping,
                            thisReadyCovers, maxCoverLen);
                    });
            });

            q_ct1.wait();
        }

        // int nEntries = reconstructHashTable.retrieve_all(vReconstructedKeys, vReconstructedIds, nObjs, 1);
        // printf("  Total number of AND nodes: %d\n", nEntries);

        // call getReadyCovers for next step
        // nReadyCovers = getReadyCovers(
        //     vCoverNodeIdMapping, vNodeCoverIdMapping, vCoverTable, vCoverTableLinks, vCoverTableLens, 
        //     vCoverNodeNewLitMapping, vThisIterReady, vGatheredReadyCovers, nCovers, nPIs
        // );
        levelCount--;
    }
    printf("Reconstruct complete! #reconstructed covers = %d\n", nReconstructed);
    assert(nReconstructed == nCovers);

    // 2.4 reconstruct complete. retrieve all nodes in reconstruct hashtable.

    // retrieved results are already sorted in ids. get new nAnds.
    int nEntries = reconstructHashTable.retrieve_all(vReconstructedKeys, vReconstructedIds, nObjs, 1);

    // 2.5 do re-id

    // generate consecutive new ids starting from nPIs + 1, and save into hash table
    HashTable<uint32, uint32> newToOutId((int)(nEntries / HT_LOAD_FACTOR));
    uint32 * htOutKeys = newToOutId.get_keys_storage();
    uint32 * htOutValues = newToOutId.get_values_storage();
    int htOutCapacity = newToOutId.get_capacity();

    nsycl::iota(q_ct1, vOutIds,
               vOutIds + nEntries, nPIs + 1, 1);
    newToOutId.insert_batch_no_update(vReconstructedIds, vOutIds, nEntries);

    int nObjsNew = nEntries + nPIs + 1;
    int * vFanin0New, * vFanin1New, * vNumFanoutsNew;
    vFanin0New =
                                   sycl::malloc_device<int>(nObjsNew, q_ct1);
    vFanin1New =
                                   sycl::malloc_device<int>(nObjsNew, q_ct1);
    vNumFanoutsNew =
                                   sycl::malloc_device<int>(nObjsNew, q_ct1);
    q_ct1.memset(vNumFanoutsNew, 0, nObjsNew * sizeof(int)).wait();

    // PI fanin info should be the same as before
    q_ct1.memcpy(vFanin0New, d_pFanin0, (nPIs + 1) * sizeof(int));
    q_ct1.memcpy(vFanin1New, d_pFanin1, (nPIs + 1) * sizeof(int));
    q_ct1.wait();

    // parse and convert reconstructed keys to fanins in output ids
    q_ct1.submit([&](sycl::handler &cgh) {
        auto vFanin0New_nPIs_ct4 = vFanin0New + nPIs + 1;
        auto vFanin1New_nPIs_ct5 = vFanin1New + nPIs + 1;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(nEntries, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                parseOutputRes(vReconstructedKeys, htOutKeys, htOutValues,
                               htOutCapacity, vFanin0New_nPIs_ct4,
                               vFanin1New_nPIs_ct5, vNumFanoutsNew, nEntries,
                               nPIs);
            });
    });

    // printCoverTable<<<1,1>>>(vCoverTable, vCoverTableLinks, vCoverTableLens);
    q_ct1.wait();

    // get new po literals
    q_ct1.submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, NUM_BLOCKS(nPOs, THREAD_PER_BLOCK)) *
                    sycl::range<3>(1, 1, THREAD_PER_BLOCK),
                sycl::range<3>(1, 1, THREAD_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                processPO(d_pOuts, vNodeCoverIdMapping, vCoverNodeNewLitMapping,
                          htOutKeys, htOutValues, htOutCapacity, vNewOuts,
                          vNumFanoutsNew, nPOs, nPIs, stream_ct1);
            });
    });
    q_ct1.wait();

    printf("#nodes = %d\n", nEntries);

#ifdef DEBUG
    int * resultFanin0, * resultFanin1, * resultNumFanouts, * resultPOs;
    resultFanin0 = (int *) malloc(nObjsNew * sizeof(int));
    resultFanin1 = (int *) malloc(nObjsNew * sizeof(int));
    resultNumFanouts = (int *) malloc(nObjsNew * sizeof(int));
    resultPOs = (int *) malloc(nPOs * sizeof(int));

    cudaMemcpy(resultFanin0, vFanin0New, nObjsNew * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultFanin1, vFanin1New, nObjsNew * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultNumFanouts, vNumFanoutsNew, nObjsNew * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultPOs, vNewOuts, nPOs * sizeof(int), cudaMemcpyDeviceToHost);

    printResults(resultFanin0, resultFanin1, resultNumFanouts, resultPOs, nEntries, nPIs, nPOs);
    free(resultFanin0);
    free(resultFanin1);
    free(resultNumFanouts);
    free(resultPOs);
#endif

    // free all allocated device memory
    sycl::free(vNodes, q_ct1);
    sycl::free(vNodes2, q_ct1);
    sycl::free(vNodesFilter, q_ct1);
    sycl::free(vCoverWiseLens, q_ct1);
    sycl::free(vCoverRanges, q_ct1);
    sycl::free(vNodesStatus, q_ct1);
    sycl::free(vNodesIndices, q_ct1);

    sycl::free(vCanonTable, q_ct1);
    sycl::free(vCoverTable, q_ct1);
    sycl::free(vCoverTableLinks, q_ct1);
    sycl::free(vCoverTableLens, q_ct1);
    sycl::free(nCoverTableNext, q_ct1);
    sycl::free(vLastAppearLevel, q_ct1);
    sycl::free(vLastAppearLevelNodes, q_ct1);
    sycl::free(vLastAppearLevelRanges, q_ct1);
    free(levelRanges);

    sycl::free(vNodeCoverIdMapping, q_ct1);
    sycl::free(vCoverNodeIdMapping, q_ct1);
    sycl::free(vCoverNodeNewLitMapping, q_ct1);
    sycl::free(vThisIterReady, q_ct1);
    sycl::free(vGatheredReadyCovers, q_ct1);
    sycl::free(vLocalReconstructArrays, q_ct1);
    sycl::free(vLocalReconstructLevels, q_ct1);
    sycl::free(vLocalReconstructLens, q_ct1);
    sycl::free(dReduce, q_ct1);
    sycl::free(vStepNodeKeys, q_ct1);
    sycl::free(vStepNodeValues, q_ct1);
    sycl::free(vStepNodeMask, q_ct1);
    sycl::free(vStepNodeLevels, q_ct1);

    sycl::free(vReconstructedKeys, q_ct1);
    sycl::free(vReconstructedIds, q_ct1);
    sycl::free(vOutIds, q_ct1);

    auto phase2_t = hrClock();
    printf("Phase 2 time: %lf\n", (phase2_t - phase2_start_t) / (double) NS_PER_SEC);
    printf("Total time: %lf\n", (phase2_t - start_t) / (double) NS_PER_SEC);

    return {vFanin0New, vFanin1New, vNumFanoutsNew, vNewOuts, nEntries};
}

void printResults(const int * resultFanin0, const int * resultFanin1, const int * resultNumFanouts, const int * resultPOs,
                  const int len, const int nPIs, const int nPOs) {
    printf("-------Balanced AIG-------\n");
    printf("id\tfanin0\tfanin1\tnumFanouts\n");
    for (int i = 0; i <= nPIs; i++) {
        printf("%d\t\t\t%d\n", i, resultNumFanouts[i]);
    }
    for (int i = nPIs + 1; i < len + nPIs + 1; i++) {
        int lit1 = resultFanin0[i];
        int lit2 = resultFanin1[i];

        printf("%d\t", i);
        printf("%s%d\t", (lit1 & 1) ? "!" : "", lit1 >> 1);
        printf("%s%d\t", (lit2 & 1) ? "!" : "", lit2 >> 1);
        printf("%d\n", resultNumFanouts[i]);
    }
    for (int i = 0; i < nPOs; i++) {
        int lit = resultPOs[i];
        int id = lit >> 1;
        printf("%s%d\n", (lit & 1) ? "!" : "", id);
    }
    printf("#nodes = %d\n", len);
}

#ifdef DEBUG
__global__ void printLevelInfo(int * vNodes, int * vNodesFilter, int levelCount, int len, int nPIs, int nPOs) {
    printf("---Level %d global input list [len = %d]---\n", levelCount, len);
    int lit;
    for (int i = 0; i < len; i++) {
        lit = vNodes[i];
        printf("%s%d\t", dUtils::AigNodeIsComplement(lit) ? "!" : "", dUtils::AigNodeIDDebug(lit, nPIs, nPOs));
    }
    printf("\n");
    printf("---Level %d filter---\n", levelCount);
    for (int i = 0; i < len; i++) {
        printf("%d\t", vNodesFilter[i]);
    }
    printf("\n");
}

void printLevelFilterStats(int * vNodesFilter, int levelCount, int len) {
    int pi = 0, redundant = 0, unique = 0;
    pi = thrust::count(thrust::device, vNodesFilter, vNodesFilter + len, 0);
    redundant = thrust::count(thrust::device, vNodesFilter, vNodesFilter + len, -1);
    unique = thrust::count(thrust::device, vNodesFilter, vNodesFilter + len, 1);

    assert(pi + redundant + unique == len);
    printf("Level %d filter: PI %.2f, redundant %.2f, unique %.2f\n", levelCount, 
          (double)pi / len * 100, (double)redundant / len * 100, (double)unique / len * 100);
}

__global__ void printArray(const int * array, const int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < len; i++) {
            printf("%d ", array[i]);
        }
        printf("\n");
    }
}

__global__ void printLocalReconstructArray(int * vLocalReconstructArrays, int * vLocalReconstructLens, 
                                           const int * vLocalReconstructLevels, const int nReadyCovers,
                                           const int maxCoverLen, int nPIs, int nPOs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lit;
    if (idx == 0) {
        for (int i = 0; i < nReadyCovers; i++) {
            printf("    Cover: ");
            for (int j = 0; j < vLocalReconstructLens[i]; j++) {
                lit = vLocalReconstructArrays[i * maxCoverLen + j];
                printf("%s%d,%d ", dUtils::AigNodeIsComplement(lit) ? "!" : "", 
                       dUtils::AigNodeIDDebug(lit, nPIs, nPOs), vLocalReconstructLevels[i * maxCoverLen + j]);
            }
            printf("\n");
        }
    }
}

__global__ void printCoverTable(const int * vCoverTable, const int * vCoverTableLinks, const int * vCoverTableLens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int checklist[] = {480379, 480394, 480396};
    if (idx == 0) {
        for (int nodeId : checklist) {
            printf("  Cover of %d: ", nodeId);
            int length = vCoverTableLens[nodeId];
            int currRowIdx = nodeId;
            int columnPtr = 0;
            int currLit, currId;
            for (int i = 0; i < length; i++) {
                if (columnPtr == COVER_TABLE_NUM_COLS) {
                    columnPtr = 0;
                    currRowIdx = vCoverTableLinks[currRowIdx];
                }
                currLit = vCoverTable[currRowIdx * COVER_TABLE_NUM_COLS + (columnPtr++)];
                currId = dUtils::AigNodeID(currLit);

                printf("%s%d ", dUtils::AigNodeIsComplement(currLit) ? "!" : "", currId);
            }
            printf("\n");
        }
    }
}

#endif


