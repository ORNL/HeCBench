#include <sycl/sycl.hpp>
#include "refactor.h"
#include "hash_table.h"
#include "vectors.hpp"
#include "truth_utils.hpp"
#include "minato_isop.hpp"
#include "alg_factor.hpp"

using namespace sop;

int evaluateSubg(int rootId, int *pNewRootLevel, const int *vCuts, int nVars,
                 int nNodeMax, int nLevelMax, const int *pLevels,
                 const uint64 *htKeys, const uint32 *htValues, int htCapacity,
                 const subgUtil::Subg<SUBG_CAP> *subg) {
    int i, counter, temp;
    int lit0, lit1, id0, id1, func0, func1, currId, fCompRoot;
    uint64 key;
    uint32 retrId;
    int vFuncs[SUBG_CAP], vLevels[SUBG_CAP];

    // check the case of the resyned cut is a const or a single var of cut nodes
    if (subg->nSize == nVars + 1) {
        subgUtil::unbindAndNodeKeyFlag(subg->pArray[nVars], &lit0, &lit1, &fCompRoot);
        if (lit0 == lit1)
            return 0;
    }
    // initialize funcs (ids) and levels for the leaves
    for (i = 0; i < nVars; i++) {
        vFuncs[i] = vCuts[i];
        vLevels[i] = pLevels[vCuts[i]];
    }
    
    counter = 0;
    for (i = nVars; i < subg->nSize; i++) {
        subgUtil::unbindAndNodeKeyFlag(subg->pArray[i], &lit0, &lit1, &fCompRoot);
        assert(lit0 < lit1);
        id0 = dUtils::AigNodeID(lit0), id1 = dUtils::AigNodeID(lit1);
        assert(id0 < i && id1 < i);

        func0 = vFuncs[id0], func1 = vFuncs[id1]; // ids of its children in the original graph
        if (func0 != -1 && func1 != -1) {
            // if they are both present, find the resulting node in hashtable
            func0 = dUtils::AigNodeLitCond(func0, dUtils::AigNodeIsComplement(lit0));
            func1 = dUtils::AigNodeLitCond(func1, dUtils::AigNodeIsComplement(lit1));
            if (func0 > func1)
                temp = func0, func0 = func1, func1 = temp;
            key = formAndNodeKey(func0, func1);
            retrId = retrieve_single<uint64, uint32>(htKeys, htValues, key, htCapacity);
            if (retrId == rootId)
                return -1;
            
            if (retrId == (HASHTABLE_EMPTY_VALUE<uint64, uint32>))
                currId = -1;
            else
                currId = (int)retrId;
        } else
            currId = -1;
        
        // count the number of added nodes
        if (currId == -1) {
            if (++counter > nNodeMax)
                return -1;
        }
        // count the number of new levels
        vLevels[i] = 1 + sycl::max(vLevels[id0], vLevels[id1]);
        if (vLevels[i] > nLevelMax)
            return -1;
        // save func
        vFuncs[i] = currId;
    }
    *pNewRootLevel = vLevels[subg->nSize - 1];
    return counter;
}


// for debug checking
void getSubgTruth(subgUtil::Subg<SUBG_CAP> * subg, const int * vCuts, int nVars, unsigned * vTruth, 
                             const unsigned * vTruthElem, int nMaxCutSize) {
    int nWords = dUtils::TruthWordNum(nVars);
    int nWordsElem = dUtils::TruthWordNum(nMaxCutSize);
    int nIntNodes = subg->nSize - nVars;
    int lit0, lit1, fCompRoot;
    int fanin0Id, fanin1Id;

    unsigned *vTruthMem =
        (unsigned *)malloc(subg->nSize * nWords * sizeof(unsigned));

    // collect elementary truth tables for the cut nodes
    for (int i = 0; i < nVars; i++) {
        for (int j = 0; j < nWords; j++)
            vTruthMem[i * nWords + j] = vTruthElem[i * nWordsElem + j];
    }

    if (nIntNodes == 1) {
        subgUtil::unbindAndNodeKeyFlag(subg->pArray[subg->nSize - 1], &lit0, &lit1, &fCompRoot);
        if (lit0 == lit1) {
            assert(fCompRoot == dUtils::AigNodeIsComplement(lit0));
            fanin0Id = dUtils::AigNodeID(lit0);
            int cutIdx = 0;
            while (cutIdx < nVars && vCuts[cutIdx] != fanin0Id) cutIdx++;
            if (cutIdx == nVars) {
                assert(fanin0Id == 0);
                if (lit0 == dUtils::AigConst0)
                    truthUtil::clear(vTruth, nVars);
                else
                    truthUtil::fill(vTruth, nVars);
                return;
            }
            assert(cutIdx < nVars);

            for (int j = 0; j < nWords; j++)
                vTruth[j] = vTruthMem[cutIdx * nWords + j];
            if (fCompRoot)
                truthUtil::truthNot(vTruth, vTruth, nVars);
            return;
        }
    }

    for (int i = nVars; i < subg->nSize; i++) {
        subgUtil::unbindAndNodeKeyFlag(subg->pArray[i], &lit0, &lit1, &fCompRoot);
        fanin0Id = dUtils::AigNodeID(lit0), fanin1Id = dUtils::AigNodeID(lit1);
        assert(fanin0Id < i && fanin1Id < i);

        if (!dUtils::AigNodeIsComplement(lit0) && !dUtils::AigNodeIsComplement(lit1))
            for (int j = 0; j < nWords; j++)
                vTruthMem[i * nWords + j] = vTruthMem[fanin0Id * nWords + j] & vTruthMem[fanin1Id * nWords + j];
        else if (!dUtils::AigNodeIsComplement(lit0) && dUtils::AigNodeIsComplement(lit1))
            for (int j = 0; j < nWords; j++)
                vTruthMem[i * nWords + j] = vTruthMem[fanin0Id * nWords + j] & ~vTruthMem[fanin1Id * nWords + j];
        else if (dUtils::AigNodeIsComplement(lit0) && !dUtils::AigNodeIsComplement(lit1))
            for (int j = 0; j < nWords; j++)
                vTruthMem[i * nWords + j] = ~vTruthMem[fanin0Id * nWords + j] & vTruthMem[fanin1Id * nWords + j];
        else
            for (int j = 0; j < nWords; j++)
                vTruthMem[i * nWords + j] = ~vTruthMem[fanin0Id * nWords + j] & ~vTruthMem[fanin1Id * nWords + j];
    }
    for (int j = 0; j < nWords; j++)
        vTruth[j] = vTruthMem[(subg->nSize - 1) * nWords + j];
    if (fCompRoot)
        truthUtil::truthNot(vTruth, vTruth, nVars);

    free(vTruthMem);
}

SYCL_EXTERNAL void resynCut(const int *vResynInd, const int *vCutTable,
                            const int *vCutSizes, const int *vNumSaved,
                            const uint64 *htKeys, const uint32 *htValues,
                            int htCapacity, const int *pLevels,
                            uint64 *vSubgTable, int *vSubgLinks, int *vSubgLens,
                            int *pSubgTableNext, unsigned *vTruth,
                            const int *vTruthRanges, const unsigned *vTruthElem,
                            int nMaxCutSize, int nResyn) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int nThreads = item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    int idx = item_ct1.get_global_id(2);
    int rootId;
    int nVars, nWords, nSaved, nAdded0, nAdded1;
    int nNewLevel0, nNewLevel1;
    int startIdx, endIdx;
    VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> vecsMem;
    subgUtil::Subg<SUBG_CAP> subg0, subg1;
    subgUtil::Subg<SUBG_CAP> * pSubg;
    int fSelectedSubg;

    for (; idx < nResyn; idx += nThreads) {
        rootId = vResynInd[idx];
        nVars = vCutSizes[rootId];
        nSaved = vNumSaved[rootId];
        nWords = dUtils::TruthWordNum(nVars);
        fSelectedSubg = -1;

        startIdx = (idx == 0 ? 0 : vTruthRanges[idx - 1]);
        endIdx = vTruthRanges[idx];
        assert(endIdx - startIdx == nWords);

        // isop + factor
        // printf("Root id: %d; cut nodes: ", rootId);
        // for (int i = 0; i < nVars; i++)
        //     printf("%d ", vCutTable[rootId * CUT_TABLE_SIZE + i]);
        // printf("\n");

        // if (truthUtil::isConst0(vTruth + startIdx, nVars))
        //     printf(" ** encountered const 0 truth table!\n");
        // if (truthUtil::isConst1(vTruth + startIdx, nVars))
        //     printf(" ** encountered const 1 truth table!\n");

        minatoIsop(vTruth + startIdx, nVars, &vecsMem);
        sopFactor(vecsMem.pArray, vecsMem.nSize, 0, &vCutTable[rootId * CUT_TABLE_SIZE], nVars, &vecsMem, &subg0);
        nAdded0 = evaluateSubg(rootId, &nNewLevel0, &vCutTable[rootId * CUT_TABLE_SIZE], 
                              nVars, nSaved, 1000000000, pLevels, htKeys, htValues, htCapacity, &subg0);
        if (nAdded0 > -1) {
            fSelectedSubg = 0;
        }

        // check isop + factor in the complemented case
        truthUtil::truthNot(vTruth + startIdx, vTruth + startIdx, nVars);
        minatoIsop(vTruth + startIdx, nVars, &vecsMem);
        sopFactor(vecsMem.pArray, vecsMem.nSize, 1, &vCutTable[rootId * CUT_TABLE_SIZE], nVars, &vecsMem, &subg1);
        nAdded1 = evaluateSubg(rootId, &nNewLevel1, &vCutTable[rootId * CUT_TABLE_SIZE], 
                              nVars, nSaved, 1000000000, pLevels, htKeys, htValues, htCapacity, &subg1);
        if (nAdded1 > -1) {
            if (nAdded0 == -1)
                fSelectedSubg = 1;
            else {
                if (nAdded1 < nAdded0 || (nAdded1 == nAdded0 && (subg1.isConst() || nNewLevel1 < nNewLevel0)))
                    fSelectedSubg = 1;
            }
        }
        
        // printf("  fSelectedSubg = %d, size0 = %d, size1 = %d\n", fSelectedSubg, subg0.nSize - nVars, subg1.nSize - nVars);
        
        // copy the selected subgraph into the global table
        // NOTE remember to only copy subg[nVars:] as the final subgraph result
        if (fSelectedSubg == -1) {
            // printf(" REJECT subgraph since the original structure is better\n");
            vSubgLens[idx] = 0;
            continue;
        }
        pSubg = (fSelectedSubg == 0 ? &subg0 : &subg1);
        vSubgLens[idx] = pSubg->nSize - nVars;
        assert(vSubgLens[idx] > 0);
        assert(vSubgLinks[idx] == -1);
        // printf("  NEW subgraph with %d nodes, adding %d, removing original structure saves %d nodes\n", 
        //        vSubgLens[idx], fSelectedSubg == 0 ? nAdded0 : nAdded1, nSaved);

        int currRowIdx, lastRowIdx, columnPtr;
        currRowIdx = idx, columnPtr = 0;
        vSubgLinks[currRowIdx] = 0;
        for (int i = nVars; i < pSubg->nSize; i++) {
            if (columnPtr == SUBG_TABLE_SIZE) {
                // expand a new row
                lastRowIdx = currRowIdx;
                currRowIdx = nsycl::atomic_fetch_add(pSubgTableNext,
                                                                1);
                assert(currRowIdx < 2 * nResyn - 1);
                assert(vSubgLinks[currRowIdx] == -1);
                
                vSubgLinks[lastRowIdx] = currRowIdx;
                vSubgLinks[currRowIdx] = 0;
                columnPtr = 0;
            }
            vSubgTable[currRowIdx * SUBG_TABLE_SIZE + (columnPtr++)] = pSubg->pArray[i];
        }

        // debug
        // unsigned * vTruthTemp = (unsigned *) malloc(nWords * sizeof(unsigned));
        // getSubgTruth(pSubg, &vCutTable[rootId * CUT_TABLE_SIZE], nVars, vTruthTemp, vTruthElem, nMaxCutSize);
        // truthUtil::truthNot(vTruth + startIdx, vTruth + startIdx, nVars);
        // assert(truthUtil::truthEqual(vTruth + startIdx, vTruthTemp, nVars));
        // free(vTruthTemp);
    }
}

SYCL_EXTERNAL void factorFromTruth(const int *vCuts, const int *vCutRanges,
                                   uint64 *vSubgTable, int *vSubgLinks,
                                   int *vSubgLens, int *pSubgTableNext,
                                   const unsigned *vTruth,
                                   const unsigned *vTruthNeg,
                                   const int *vTruthRanges,
                                   const unsigned *vTruthElem, int nResyn) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    int nVars;
    int cutStartIdx, cutEndIdx, truthStartIdx, truthEndIdx;
    int fNeg = 0;
    VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> vecsMem;
    subgUtil::Subg<SUBG_CAP> subg;

    // the number of threads launched should be 2 * nResyn
    if (idx < 2 * nResyn) {
        if (idx >= nResyn) {
            idx -= nResyn;
            fNeg = 1;
        }

        cutStartIdx = (idx == 0 ? 0 : vCutRanges[idx - 1]);
        cutEndIdx = vCutRanges[idx];
        nVars = cutEndIdx - cutStartIdx;

        truthStartIdx = (idx == 0 ? 0 : vTruthRanges[idx - 1]);
        truthEndIdx = vTruthRanges[idx];
        assert(truthEndIdx - truthStartIdx == dUtils::TruthWordNum(nVars));

        const unsigned * pTruth = (fNeg ? vTruthNeg + truthStartIdx : vTruth + truthStartIdx);

        // isop + factor
        minatoIsop(pTruth, nVars, &vecsMem);
        // NOTE: the original __syncthreads() here is removed. It sits in
        // divergent control flow (only idx < 2*nResyn threads enter this block),
        // which deadlocks the work-group on the Level-Zero backend. The barrier
        // is functionally unnecessary: each thread processes its own cone with
        // private vecsMem/subg and shares no state with other threads.
        sopFactor(vecsMem.pArray, vecsMem.nSize, fNeg, vCuts + cutStartIdx, nVars, &vecsMem, &subg);
        // (removed divergent __syncthreads(); see note above)

        // save synthesized graph into global table
        int currRowIdx, lastRowIdx, columnPtr;
        currRowIdx = 2 * idx + fNeg; // corresponding pairs are consecutively stored
        columnPtr = 0;

        vSubgLens[currRowIdx] = subg.nSize - nVars;
        assert(vSubgLens[currRowIdx] > 0);
        assert(vSubgLinks[currRowIdx] == -1);
        vSubgLinks[currRowIdx] = 0;
        for (int i = nVars; i < subg.nSize; i++) {
            if (columnPtr == SUBG_TABLE_SIZE) {
                // expand a new row
                lastRowIdx = currRowIdx;
                currRowIdx = nsycl::atomic_fetch_add(pSubgTableNext,
                                                                1);
                assert(currRowIdx < 4 * nResyn - 1);
                assert(vSubgLinks[currRowIdx] == -1);
                
                vSubgLinks[lastRowIdx] = currRowIdx;
                vSubgLinks[currRowIdx] = 0;
                columnPtr = 0;
            }
            vSubgTable[currRowIdx * SUBG_TABLE_SIZE + (columnPtr++)] = subg.pArray[i];
        }
        // (removed divergent __syncthreads(); see note above)

        // unsigned * vTruthTemp = (unsigned *) malloc(dUtils::TruthWordNum(nVars) * sizeof(unsigned));
        // getSubgTruth(&subg, vCuts + cutStartIdx, nVars, vTruthTemp, vTruthElem, 12);
        // if (fNeg)
        //     truthUtil::truthNot(vTruthTemp, vTruthTemp, nVars);
        // assert(truthUtil::truthEqual(pTruth, vTruthTemp, nVars));
        // free(vTruthTemp);
    }
}


