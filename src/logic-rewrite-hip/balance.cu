#include <vector>
#include <unordered_map>
#include <tuple>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include "common.h"
#include "hash_table.h"
#include "balance.h"

const int MAX_LOCAL_COVER_SIZE = 1024;
const int DFS_COVER_STACK_SIZE = 128;
const int COVER_TABLE_NUM_COLS = 4;
const double HT_LOAD_FACTOR = 0.5;

// managed memory
__managed__ int isEnd;

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

__global__ void checkCoverTravEnd(int * isEnd, int * vInputs, int * vInputsFilter, 
                                  int * vCanonTable, const int nPIs, const int arrayLen) {
    /**
     * This is a full-filtering marking both "PI" and "R" (in-level redundancy). 
     **/

    // isEnd should be 1 when passed in!
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
            int old = atomicCAS(&vCanonTable[nodeId], 0, idx + 1);
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









__device__ uint32 retrieveAndNode(int lit1, int lit2,
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

__global__ void prepareDataToInsert(int * vLocalReconstructArrays, int * vLocalReconstructLevels, int * vLocalReconstructLens, 
                                    uint64 * vStepNodeKeys, uint32 * vStepNodeValues, int * vStepNodeMask, uint32 * vStepNodeLevels,
                                    const uint64 * htReconstructKeys, const uint32 * htReconstructValues, const int htReconstructCapacity,
                                    const uint32 * htLevelKeys, const uint32 * htLevelValues, const int htLevelCapacity,
                                    const int newIdCounter, const int nReadyCovers, const int nPIs, const int nPOs, const int maxCoverLen) {
    /**
     * Prepare one batch of AND node to be inserted to hash table. 
     * 
     * Result of this kernel function:
     * vStepNodeMask = 0 (already complete including constant false, invalid entry): vStepNodeKeys, vStepNodeValues, vStepNodeLevels are all invalid
     * vStepNodeMask = 1 (non-existing nodes): vStepNodeKeys, vStepNodeValues, vStepNodeLevels are all valid
     * vStepNodeMask = 2 (existing nodes): vStepNodeValues, vStepNodeLevels are valid, but vStepNodeKeys are invalid
     * */
    // newIdCounter indicates the max id in reconstruct hashtable + 1, i.e., first available new id.

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
        vStepNodeKeys[idx] = formAndNodeKey(lit1, lit2);
        vStepNodeValues[idx] = (uint32)(newIdCounter + idx); // save new id, instead of literal as for mask = 2 nodes
        vStepNodeLevels[idx] = (uint32)max(lv1, lv2) + 1;
    }
}


__global__ void sharedNodeDrivenPermute(int * vLocalReconstructArrays, int * vLocalReconstructLevels, int * vLocalReconstructLens, 
                                        const uint64 * htReconstructKeys, const uint32 * htReconstructValues, const int htReconstructCapacity,
                                        const int nReadyCovers, const int maxCoverLen) {
    // this can be optional. 
    // In original ABC we tested that without this step, num of node reduction dropped around 2.5%.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

void getTopTwoNodes(int * vLocalReconstructArrays, int * vLocalReconstructLevels, int * vLocalReconstructLens, 
                    const uint64 * htReconstructKeys, const uint32 * htReconstructValues, const int htReconstructCapacity, 
                    const uint32 * htLevelKeys, const uint32 * htLevelValues, const int htLevelCapacity,
                    uint64 * vStepNodeKeys, uint32 * vStepNodeValues, int * vStepNodeMask, uint32 * vStepNodeLevels,
                    const int newIdCounter, const int nReadyCovers, const int nPIs, const int nPOs, const int maxCoverLen) {
    // permute local reconstruct arrays, get top two for each cover, 
    // prepare them in hashtable key format and form 
    // 1) consecutive arrays of keys and values; 2) a mask indicating whether is null or trivial cases; 3) new node levels.
    sharedNodeDrivenPermute<<<NUM_BLOCKS(nReadyCovers, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vLocalReconstructArrays, vLocalReconstructLevels, vLocalReconstructLens, 
        htReconstructKeys, htReconstructValues, htReconstructCapacity, nReadyCovers, maxCoverLen
    );
    hipDeviceSynchronize();
    prepareDataToInsert<<<NUM_BLOCKS(nReadyCovers, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vLocalReconstructArrays, vLocalReconstructLevels, vLocalReconstructLens, 
        vStepNodeKeys, vStepNodeValues, vStepNodeMask, vStepNodeLevels, 
        htReconstructKeys, htReconstructValues, htReconstructCapacity, 
        htLevelKeys, htLevelValues, htLevelCapacity, newIdCounter, nReadyCovers, nPIs, nPOs, maxCoverLen
    );
    hipDeviceSynchronize();

}

__global__ void prepareReconstructArrays(const int * vGatheredReadyCovers, const int * vNodeCoverIdMapping, 
                                         const int * vCoverNodeIdMapping, const int * vCoverNodeNewLitMapping, 
                                         const int * vCoverTable, const int * vCoverTableLinks, const int * vCoverTableLens, 
                                         const uint32 * htLevelKeys, const uint32 * htLevelValues, const int htLevelCapacity,
                                         int * vLocalReconstructArrays, int * vLocalReconstructLevels, int * vLocalReconstructLens, 
                                         const int nReadyCovers, const int nPIs, const int maxCoverLen,
                                         const int sortDecId) {
    // gather cover input list from global cover table to local reconstruct arrays,
    // then do sorting
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
        thrust::default_random_engine rng(idx);
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

        if (sortDecId) {
            thrust::sort(
                thrust::seq, 
                thrust::make_zip_iterator(thrust::make_tuple(&vLocalReconstructArrays[localArrayStartIdx], 
                                          &vLocalReconstructLevels[localArrayStartIdx])), 
                thrust::make_zip_iterator(thrust::make_tuple(&vLocalReconstructArrays[localArrayStartIdx + length], 
                                          &vLocalReconstructLevels[localArrayStartIdx + length])), 
                dUtils::decreaseLevelIds<int, int>()
            );

            // thrust::sort(
            //     thrust::seq, 
            //     thrust::make_zip_iterator(thrust::make_tuple(&vLocalReconstructArrays[localArrayStartIdx], 
            //                               &vLocalReconstructLevels[localArrayStartIdx],
            //                               &vRand[0])), 
            //     thrust::make_zip_iterator(thrust::make_tuple(&vLocalReconstructArrays[localArrayStartIdx + length], 
            //                               &vLocalReconstructLevels[localArrayStartIdx + length],
            //                               &vRand[length])), 
            //     dUtils::decreaseLevelsPerm<int, int>()
            // );
        } else {
            thrust::shuffle(thrust::seq, 
                        thrust::make_zip_iterator(thrust::make_tuple(&vLocalReconstructArrays[localArrayStartIdx], 
                                          &vLocalReconstructLevels[localArrayStartIdx])), 
                        thrust::make_zip_iterator(thrust::make_tuple(&vLocalReconstructArrays[localArrayStartIdx + length], 
                                          &vLocalReconstructLevels[localArrayStartIdx + length])), 
                        rng);

            thrust::sort(
                thrust::seq, 
                thrust::make_zip_iterator(thrust::make_tuple(&vLocalReconstructArrays[localArrayStartIdx], 
                                          &vLocalReconstructLevels[localArrayStartIdx])), 
                thrust::make_zip_iterator(thrust::make_tuple(&vLocalReconstructArrays[localArrayStartIdx + length], 
                                          &vLocalReconstructLevels[localArrayStartIdx + length])), 
                dUtils::decreaseLevels<int, int>()
            );
        }
        

        
    }
}


__global__ void addBackLocalArrays(int * vLocalReconstructArrays, int * vLocalReconstructLevels, int * vLocalReconstructLens, 
                                   const uint32 * vStepNodeValues, const int * vStepNodeMask, const uint32 * vStepNodeLevels,
                                   const int nReadyCovers, const int maxCoverLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // mask = 0 means that idx has complete reconstruction in previous steps
    if (idx < nReadyCovers && vStepNodeMask[idx] != 0) {
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

__global__ void recordReconstructedCovers(const int * vLocalReconstructArrays, const int * vLocalReconstructLens, 
                                          const int * vGatheredReadyCovers, const int * vNodeCoverIdMapping, int * vCoverNodeNewLitMapping, 
                                          const int nReadyCovers, const int maxCoverLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void genReadyMask(const int * vCoverNodeIdMapping, const int * vNodeCoverIdMapping, 
                             const int * vCoverTable, const int * vCoverTableLinks, const int * vCoverTableLens, 
                             const int * vCoverNodeNewLitMapping, int * vThisIterReady, const int arrayLen, const int nPIs) {
    // vCoverNodeNewLitMapping indicates ready or not
    // generate a mask vThisIterReady for all covers that can be re-constructed in this iter
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void gatherByScannedMask(int * vScannedMask, int * vGathered, const int arrayLen, const int idOffset=0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

int gatherByMask(int * vMask, int * vGathered, const int arrayLen, const int idOffset) {
    // gathered results will be saved in vGathered, make sure vGathered has at least arrayLen entries
    // contents in vMask will be modified
    int nGathered = -1;

    thrust::inclusive_scan(thrust::device, vMask, vMask + arrayLen, vMask);
    hipDeviceSynchronize();
    hipMemcpy(&nGathered, &vMask[arrayLen - 1], sizeof(int), hipMemcpyDeviceToHost);
    assert(nGathered != -1);

    if (nGathered > 0) {
        gatherByScannedMask<<<NUM_BLOCKS(arrayLen, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vMask, vGathered, arrayLen, idOffset
        );
    }
    return nGathered;
}

int getReadyCovers(const int * vCoverNodeIdMapping, const int * vNodeCoverIdMapping, 
                   const int * vCoverTable, const int * vCoverTableLinks, const int * vCoverTableLens, 
                   const int * vCoverNodeNewLitMapping, int * vThisIterReady, int * vGatheredReadyCovers, 
                   const int nCovers, const int nPIs) {
    // get all ready cover ids, and the total number of ready covers
    int nReadyCovers = -1;

    genReadyMask<<<NUM_BLOCKS(nCovers + 1, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vCoverNodeIdMapping, vNodeCoverIdMapping, vCoverTable, vCoverTableLinks, vCoverTableLens, 
        vCoverNodeNewLitMapping, vThisIterReady, nCovers + 1, nPIs
    );

    nReadyCovers = gatherByMask(vThisIterReady + 1, vGatheredReadyCovers, nCovers, 1);

    return nReadyCovers;
}


__global__ void markIsCoverOutput(const int * vCoverTableLens, int * vMarks, const int arrayLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < arrayLen) {
        vMarks[idx] = (vCoverTableLens[idx] != -1 ? 1 : 0);
    }
}

__global__ void getCoverToNodeIdMapping(const int * vNodeCoverIdMapping, const int * vCoverTableLens, 
                                        int * vCoverNodeIdMapping, const int arrayLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < arrayLen) {
        if (vCoverTableLens[idx] != -1) {
            // idx is output node id that corresponds to a cover
            vCoverNodeIdMapping[vNodeCoverIdMapping[idx]] = idx;
        }
    }
}

__global__ void gatherWithFilter(int * isEnd, 
                                 const int * vCoverTable, const int * vCoverTableLinks, 
                                 const int * vNodes, const int * vCoverRanges, int * vCanonTable, 
                                 int * vNewGlobalList, int * vNodesFilter, const int nPIs, const int arrayLen) {
    // vCanonTable should be all zero before this kernel call
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
            laterThread = atomicCAS(&vCanonTable[currId], 0, 1);
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

__device__ int localCoverTravToTable(int nodeLit, const int * pFanin0, const int * pFanin1, const int * pNumFanouts, 
                                     int * vCoverTable, int * vCoverTableLinks, int * vCoverTableLens, int * nCoverTableNext,
                                     int * superLen, const int nPIs) {
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
                currRowIdx = atomicAdd(nCoverTableNext, 1);
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

__global__ void coverFindingToTable(int * vNodes, int * vNodesStatus, int * vLastAppearLevel, 
                                    const int * pFanin0, const int * pFanin1, const int * pNumFanouts, 
                                    int * vCoverTable, int * vCoverTableLinks, int * vCoverTableLens, int * nCoverTableNext,
                                    const int nPIs, const int arrayLen, const int levelCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void findLevelNodeRanges(const int * vLastAppearLevel, int * vLastAppearLevelRanges, const int nCovers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == nCovers - 1) {
        printf("*** Max Level: %d\n", vLastAppearLevel[idx]);
        vLastAppearLevelRanges[vLastAppearLevel[idx]] = idx + 1;
    } else if (idx < nCovers - 1) {
        if (vLastAppearLevel[idx] < vLastAppearLevel[idx + 1]) {
            vLastAppearLevelRanges[vLastAppearLevel[idx]] = idx + 1;
        }
    }
}

__global__ void parseOutputRes(const uint64 * vReconstructedKeys, 
                               const uint32 * htOutKeys, const uint32 * htOutValues, const int htOutCapacity,
                               int * vFanin0New, int * vFanin1New, int * vNumFanoutsNew, 
                               const int nEntries, const int nPIs) {
    // NOTE vFanin0New, vFanin1New should point to the begin of AND node storage;
    //      vNumFanoutsNew should point to the begin of PIs + AND node storage!
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
        atomicAdd(&vNumFanoutsNew[newId0], 1);
        atomicAdd(&vNumFanoutsNew[newId1], 1);
    }
}

__global__ void processPO(const int * d_pOuts, const int * vNodeCoverIdMapping, const int * vCoverNodeNewLitMapping, 
                          const uint32 * htOutKeys, const uint32 * htOutValues, const int htOutCapacity,
                          int * vNewOuts, int * vNumFanoutsNew, const int nPOs, const int nPIs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPOs) {
        int oldLit = d_pOuts[idx];
        int oldId = dUtils::AigNodeID(oldLit);

        // increase number of fanout for POs, which ensures consistency with ABC
        if (dUtils::AigIsPIConst(oldId, nPIs)) {
            vNewOuts[idx] = oldLit;
            atomicAdd(&vNumFanoutsNew[oldId], 1);
        } else {
            int newLit = vCoverNodeNewLitMapping[vNodeCoverIdMapping[oldId]];
            assert(newLit != -1);
            int outId = retrieve_single<uint32, uint32>(htOutKeys, htOutValues, dUtils::AigNodeID(newLit), htOutCapacity);
            if (outId == (HASHTABLE_EMPTY_VALUE<uint32, uint32>)) {
                // should be PI or const
                printf("oldId: %d, coverId: %d, newLit: %d\n", oldId, vNodeCoverIdMapping[oldId], newLit);
                outId = dUtils::AigNodeID(newLit);
                assert(dUtils::AigIsPIConst(outId, nPIs));
            }
            assert(outId != (HASHTABLE_EMPTY_VALUE<uint32, uint32>));

            vNewOuts[idx] = dUtils::AigNodeLitCond(outId, dUtils::AigNodeIsComplement(oldLit)^dUtils::AigNodeIsComplement(newLit));
            atomicAdd(&vNumFanoutsNew[outId], 1);
        }
    }
}

std::tuple<int *, int *, int *, int *, int> 
    balancePerformV2(int nObjs, int nPIs, int nPOs, int nNodes, 
                     int * d_pFanin0, int * d_pFanin1, int * d_pOuts, 
                     int * d_pNumFanouts, int sortDecId) {
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

    hipMalloc(&vNodes, nObjs * sizeof(int));
    hipMalloc(&vNodes2, globalListLen2 * sizeof(int));
    hipMalloc(&vNodesFilter, globalListLen2 * sizeof(int));
    hipMalloc(&vCoverWiseLens, globalListLen * sizeof(int));
    hipMalloc(&vCoverRanges, globalListLen * sizeof(int));
    hipMemcpy(vNodes, d_pOuts, nPOs * sizeof(int), hipMemcpyDeviceToDevice);
    hipMalloc(&vNodesStatus, nObjs * sizeof(int));
    hipMalloc(&vNodesIndices, nObjs * sizeof(int));

    hipMalloc(&vCanonTable, nObjs * sizeof(int));
    hipMalloc(&vLastAppearLevel, nObjs * sizeof(int));
    hipMalloc(&vLastAppearLevelNodes, nObjs * sizeof(int));
    hipMemset(vCanonTable, 0, nObjs * sizeof(int));
    hipMemset(vLastAppearLevel, -1, nObjs * sizeof(int));

    hipMalloc(&vCoverTable, 2 * nObjs * COVER_TABLE_NUM_COLS * sizeof(int));
    hipMalloc(&vCoverTableLinks, 2 * nObjs * sizeof(int));
    hipMalloc(&vCoverTableLens, nObjs * sizeof(int));
    hipMalloc(&nCoverTableNext, sizeof(int));
    hipMemset(vCoverTableLinks, -1, 2 * nObjs * sizeof(int));
    hipMemset(vCoverTableLens, -1, nObjs * sizeof(int));
    hipMemcpy(nCoverTableNext, &nObjs, sizeof(int), hipMemcpyHostToDevice);

    isEnd = 1;
    hipDeviceSynchronize();

    // len of current level global list
    int currLen = nPOs, newLen;
    // filter and convert elements in vNodes to node ids
    checkCoverTravEnd<<<NUM_BLOCKS(nPOs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        &isEnd, vNodes, vNodesFilter, vCanonTable, nPIs, nPOs
    );
    hipDeviceSynchronize();
    if (!isEnd) {
        // remove duplicates
        int * pFilteredEnd = thrust::remove_if(
            thrust::device, vNodes, vNodes + nPOs, vNodesFilter, dUtils::isNotOne<int>()
        );
        currLen = pFilteredEnd - vNodes;
    }
    if (isEnd || currLen == 0)
        return {NULL, NULL, NULL, NULL, 0};

    #ifdef DEBUG
        printLevelInfo<<<1, 1>>>(vNodes, vNodesFilter, 0, nPOs, nPIs, nPOs);
        hipDeviceSynchronize();
    #endif

    // precompute a consecutive indices array for gathering uses
    thrust::sequence(thrust::device, vNodesIndices, vNodesIndices + nObjs);

    do {
        hipMemset(vNodesStatus, 0, nObjs * sizeof(int));

        // cover findings, record into cover table and vCoverWiseLens
        coverFindingToTable<<<NUM_BLOCKS(currLen, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vNodes, vNodesStatus, vLastAppearLevel, d_pFanin0, d_pFanin1, d_pNumFanouts,
            vCoverTable, vCoverTableLinks, vCoverTableLens, nCoverTableNext, 
            nPIs, currLen, levelCount
        );
        hipDeviceSynchronize();

        // gather the indices where vNodesStatus = 1 to the next level's vNodes
        int * pNewGlobalListEnd = thrust::copy_if(
            thrust::device, vNodesIndices, vNodesIndices + nObjs,
            vNodesStatus, vNodes, dUtils::isOne<int>()
        );
        newLen = pNewGlobalListEnd - vNodes;

        // update currLen
        currLen = newLen;

        levelCount++;
        #ifdef DEBUG
            printf("Level %d, global list len %d\n", levelCount, currLen);
            printLevelInfo<<<1, 1>>>(vNodes, vNodesFilter, levelCount, currLen, nPIs, nPOs);
            printLevelFilterStats(vNodesFilter, levelCount, currLen);
            hipDeviceSynchronize();
        #endif

    } while (currLen > 0);

    hipDeviceSynchronize();
    auto phase1_t = hrClock();

    printf("Phase 1 time: %lf\n", (phase1_t - start_t) / (double) NS_PER_SEC);

    // get maximum local input length
    int * pMaxCoverLen = thrust::max_element(thrust::device, vCoverTableLens, vCoverTableLens + nObjs);
    int maxCoverLen;
    hipDeviceSynchronize();
    hipMemcpy(&maxCoverLen, pMaxCoverLen, sizeof(int), hipMemcpyDeviceToHost);
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

    hipMalloc(&vNodeCoverIdMapping, nObjs * sizeof(int));

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
    markIsCoverOutput<<<NUM_BLOCKS(nObjs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vCoverTableLens, vNodeCoverIdMapping, nObjs
    );
    // hipDeviceSynchronize();
    // note that id ranges from 1 to n, so need to allocate n + 1 entries for related data structures
    thrust::inclusive_scan(thrust::device, vNodeCoverIdMapping, vNodeCoverIdMapping + nObjs, vNodeCoverIdMapping);
    hipDeviceSynchronize();

    // get number of covers
    hipMemcpy(&nCovers, &vNodeCoverIdMapping[nObjs - 1], sizeof(int), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    printf("Total number of covers: %d\n", nCovers);
    
    // 2.1.1 compute cover ready status of each level
    hipMalloc(&vLastAppearLevelRanges, levelCount * sizeof(int));
    // assign initial node ids
    thrust::sequence(thrust::device, vLastAppearLevelNodes, vLastAppearLevelNodes + nObjs);
    int * pNewLastAppearLevelNodesEnd = thrust::remove_if(
        thrust::device, vLastAppearLevelNodes, vLastAppearLevelNodes + nObjs, vLastAppearLevel, dUtils::isMinusOne<int>()
    );
    int * pNewLastAppearLevelEnd = thrust::remove(thrust::device, vLastAppearLevel, vLastAppearLevel + nObjs, -1);
    assert(pNewLastAppearLevelNodesEnd - vLastAppearLevelNodes == nCovers);
    assert(pNewLastAppearLevelEnd - vLastAppearLevel == nCovers);

    // sort in level no.
    thrust::sort_by_key(
        thrust::device, vLastAppearLevel, vLastAppearLevel + nCovers, vLastAppearLevelNodes
    );
    // find range indices for each level
    findLevelNodeRanges<<<NUM_BLOCKS(nCovers, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(vLastAppearLevel, vLastAppearLevelRanges, nCovers);
    hipDeviceSynchronize();

    // int iterLen = 100000000;
    const int iterLen = 32768;

    int * levelRanges = (int *) malloc(levelCount * sizeof(int));
    hipMemcpy(levelRanges, vLastAppearLevelRanges, levelCount * sizeof(int), hipMemcpyDeviceToHost);

    int maxReadyCovers = levelRanges[0];
    for(int i=1; i<levelCount; i++)
        maxReadyCovers = max(maxReadyCovers, levelRanges[i] - levelRanges[i-1]);
    maxReadyCovers = min(maxReadyCovers, iterLen);

    hipMalloc(&vCoverNodeIdMapping, (nCovers + 1) * sizeof(int));
    hipMalloc(&vCoverNodeNewLitMapping, (nCovers + 1) * sizeof(int));
    hipMalloc(&vThisIterReady, (nCovers + 1) * sizeof(int));
    hipMalloc(&vGatheredReadyCovers, nCovers * sizeof(int));
    gpuErrchk( hipMalloc(&vLocalReconstructArrays, (size_t)maxReadyCovers * (size_t)maxCoverLen * sizeof(int)) );
    gpuErrchk( hipMalloc(&vLocalReconstructLevels, (size_t)maxReadyCovers * (size_t)maxCoverLen * sizeof(int)) );
    hipMalloc(&vLocalReconstructLens, nCovers * sizeof(int));
    hipMalloc(&vStepNodeKeys, nCovers * sizeof(uint64));
    hipMalloc(&vStepNodeValues, nCovers * sizeof(uint32));
    hipMalloc(&vStepNodeMask, nCovers * sizeof(int));
    hipMalloc(&vStepNodeLevels, nCovers * sizeof(uint32));
    
    hipMemset(vCoverNodeNewLitMapping, -1, (nCovers + 1) * sizeof(int));
    hipDeviceSynchronize();

    // compute inverse mapping
    getCoverToNodeIdMapping<<<NUM_BLOCKS(nObjs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vNodeCoverIdMapping, vCoverTableLens, vCoverNodeIdMapping, nObjs
    );
    hipDeviceSynchronize();

    // 2.2 outer loop: check ready covers

    uint64 * vReconstructedKeys;
    uint32 * vReconstructedIds, * vOutIds;
    int * vNewOuts;
    hipMalloc(&vReconstructedKeys, nObjs * sizeof(uint64));
    hipMalloc(&vReconstructedIds, nObjs * sizeof(uint32));
    hipMalloc(&vOutIds, nObjs * sizeof(uint32));
    hipMalloc(&vNewOuts, nPOs * sizeof(int));

    int nReadyCovers, nReconstructed = 0;
    int newIdCounter = nPIs + 1;

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
            prepareReconstructArrays<<<NUM_BLOCKS(thisReadyCovers, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
                vLastAppearLevelNodes + thisReadyStartIdx, vNodeCoverIdMapping, vCoverNodeIdMapping, vCoverNodeNewLitMapping, 
                vCoverTable, vCoverTableLinks, vCoverTableLens, 
                htLevelKeys, htLevelValues, htLevelCapacity, 
                vLocalReconstructArrays, vLocalReconstructLevels, vLocalReconstructLens,
                thisReadyCovers, nPIs, maxCoverLen, sortDecId
            );
            // hipDeviceSynchronize();

            #ifdef DEBUG
                printLocalReconstructArray<<<1, 1>>>(
                    vLocalReconstructArrays, vLocalReconstructLens, vLocalReconstructLevels, 
                    thisReadyCovers, maxCoverLen, nPIs, nPOs
                );
                hipDeviceSynchronize();
            #endif

            // 2.3 inner loop: perform one step of AND construction
            while (true) {
                // gather batch data into vStep* arrays
                assert(INT_MAX - newIdCounter > thisReadyCovers);
                getTopTwoNodes(
                    vLocalReconstructArrays, vLocalReconstructLevels, vLocalReconstructLens, 
                    htReconstructKeys, htReconstructValues, htReconstructCapacity, 
                    htLevelKeys, htLevelValues, htLevelCapacity,
                    vStepNodeKeys, vStepNodeValues, vStepNodeMask, vStepNodeLevels, 
                    newIdCounter, thisReadyCovers, nPIs, nPOs, maxCoverLen
                );
                // check whether complete or not
                int maxMask = thrust::reduce(thrust::device, vStepNodeMask, vStepNodeMask + thisReadyCovers, 0, thrust::maximum<int>());
                hipDeviceSynchronize();
                assert(maxMask >= 0 && maxMask <= 2);
                if (maxMask == 0)
                    break;
                
                // TODO (future) actually can safely remove duplicate keys before insertion. since the new ids are generated by idx, 
                //      can traceback using this value. (deal with non-deterministic)
                
                // batch insert to hashtable
                reconstructHashTable.insert_batch_no_update_masked(vStepNodeKeys, vStepNodeValues, vStepNodeMask, thisReadyCovers);
                // retrieve immediately to update values in case there are duplicate keys in this batch
                reconstructHashTable.retrieve_batch_masked(vStepNodeKeys, vStepNodeValues, vStepNodeMask, thisReadyCovers);

                // get max id in this batch of values filtered by mask, use it to update id counter
                int maxId = thrust::transform_reduce(
                    thrust::device,
                    thrust::make_zip_iterator(thrust::make_tuple(vStepNodeValues, vStepNodeMask)),
                    thrust::make_zip_iterator(thrust::make_tuple(vStepNodeValues + thisReadyCovers, vStepNodeMask + thisReadyCovers)),
                    dUtils::getValueFilteredByMask<uint32, int>(1), // 1 is true value for mask
                    0,                                              // init value
                    thrust::maximum<uint32>()
                );
                // hipDeviceSynchronize();
                if (maxId != 0) {
                    // maxId = 0 means that all values are masked out
                    newIdCounter = maxId + 1;
                }

                // for new nodes, insert the associated levels into level ht. note that if remove duplicate keys (previous TODO), 
                // this insertion can be done earlier in prepareDataToInsert.
                levelHashTable.insert_batch_no_update_masked(vStepNodeValues, vStepNodeLevels, vStepNodeMask, thisReadyCovers);

                // add result of this step back to local arrays while keeping the local array sorted w.r.t. level
                addBackLocalArrays<<<NUM_BLOCKS(thisReadyCovers, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
                    vLocalReconstructArrays, vLocalReconstructLevels, vLocalReconstructLens, 
                    vStepNodeValues, vStepNodeMask, vStepNodeLevels, thisReadyCovers, maxCoverLen
                );
                hipDeviceSynchronize();

            }

            // 2.3 add newly constructed cover output nodes into vCoverNodeNewLitMapping
            recordReconstructedCovers<<<NUM_BLOCKS(thisReadyCovers, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
                vLocalReconstructArrays, vLocalReconstructLens, 
                vLastAppearLevelNodes + thisReadyStartIdx, vNodeCoverIdMapping, vCoverNodeNewLitMapping, 
                thisReadyCovers, maxCoverLen
            );
            
            hipDeviceSynchronize();
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

    thrust::sequence(thrust::device, vOutIds, vOutIds + nEntries, nPIs + 1, 1);
    newToOutId.insert_batch_no_update(vReconstructedIds, vOutIds, nEntries);

    int nObjsNew = nEntries + nPIs + 1;
    int * vFanin0New, * vFanin1New, * vNumFanoutsNew;
    gpuErrchk( hipMalloc(&vFanin0New, sizeof(int) * nObjsNew) );
    gpuErrchk( hipMalloc(&vFanin1New, sizeof(int) * nObjsNew) );
    gpuErrchk( hipMalloc(&vNumFanoutsNew, sizeof(int) * nObjsNew) );
    hipMemset(vNumFanoutsNew, 0, nObjsNew * sizeof(int));

    // PI fanin info should be the same as before
    hipMemcpy(vFanin0New, d_pFanin0, (nPIs + 1) * sizeof(int), hipMemcpyDeviceToDevice);
    hipMemcpy(vFanin1New, d_pFanin1, (nPIs + 1) * sizeof(int), hipMemcpyDeviceToDevice);
    hipDeviceSynchronize();

    // parse and convert reconstructed keys to fanins in output ids
    parseOutputRes<<<NUM_BLOCKS(nEntries, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        vReconstructedKeys, htOutKeys, htOutValues, htOutCapacity, 
        vFanin0New + nPIs + 1, vFanin1New + nPIs + 1, vNumFanoutsNew, nEntries, nPIs
    );

    // printCoverTable<<<1,1>>>(vCoverTable, vCoverTableLinks, vCoverTableLens);
    hipDeviceSynchronize();

    // get new po literals
    processPO<<<NUM_BLOCKS(nPOs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        d_pOuts, vNodeCoverIdMapping, vCoverNodeNewLitMapping, 
        htOutKeys, htOutValues, htOutCapacity, 
        vNewOuts, vNumFanoutsNew, nPOs, nPIs
    );
    hipDeviceSynchronize();

    printf("#nodes = %d\n", nEntries);

#ifdef DEBUG
    int * resultFanin0, * resultFanin1, * resultNumFanouts, * resultPOs;
    resultFanin0 = (int *) malloc(nObjsNew * sizeof(int));
    resultFanin1 = (int *) malloc(nObjsNew * sizeof(int));
    resultNumFanouts = (int *) malloc(nObjsNew * sizeof(int));
    resultPOs = (int *) malloc(nPOs * sizeof(int));

    hipMemcpy(resultFanin0, vFanin0New, nObjsNew * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(resultFanin1, vFanin1New, nObjsNew * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(resultNumFanouts, vNumFanoutsNew, nObjsNew * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(resultPOs, vNewOuts, nPOs * sizeof(int), hipMemcpyDeviceToHost);

    printResults(resultFanin0, resultFanin1, resultNumFanouts, resultPOs, nEntries, nPIs, nPOs);
    free(resultFanin0);
    free(resultFanin1);
    free(resultNumFanouts);
    free(resultPOs);
#endif

    // free all allocated device memory
    hipFree(vNodes);
    hipFree(vNodes2);
    hipFree(vNodesFilter);
    hipFree(vCoverWiseLens);
    hipFree(vCoverRanges);
    hipFree(vNodesStatus);
    hipFree(vNodesIndices);

    hipFree(vCanonTable);
    hipFree(vCoverTable);
    hipFree(vCoverTableLinks);
    hipFree(vCoverTableLens);
    hipFree(nCoverTableNext);
    hipFree(vLastAppearLevel);
    hipFree(vLastAppearLevelNodes);
    hipFree(vLastAppearLevelRanges);
    free(levelRanges);

    hipFree(vNodeCoverIdMapping);
    hipFree(vCoverNodeIdMapping);
    hipFree(vCoverNodeNewLitMapping);
    hipFree(vThisIterReady);
    hipFree(vGatheredReadyCovers);
    hipFree(vLocalReconstructArrays);
    hipFree(vLocalReconstructLevels);
    hipFree(vLocalReconstructLens);
    hipFree(vStepNodeKeys);
    hipFree(vStepNodeValues);
    hipFree(vStepNodeMask);
    hipFree(vStepNodeLevels);

    hipFree(vReconstructedKeys);
    hipFree(vReconstructedIds);
    hipFree(vOutIds);

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


