#pragma once
#include <sycl/sycl.hpp>
#include <tuple>
#include "common.h"


std::tuple<int, int *, int *, int *, int> 
refactorPerform(bool fUseZeros, int cutSize,
                int nObjs, int nPIs, int nPOs, int nNodes, 
                const int * d_pFanin0, const int * d_pFanin1, const int * d_pOuts, 
                const int * d_pNumFanouts, const int * d_pLevel, 
                int * pOuts, int * pNumFanouts);
std::tuple<int, int *, int *, int *, int> 
refactorMFFCPerform(bool fUseZeros, int cutSize,
                    int nObjs, int nPIs, int nPOs, int nNodes, 
                    const int * d_pFanin0, const int * d_pFanin1, const int * d_pOuts, 
                    const int * d_pNumFanouts, const int * d_pLevel, 
                    int * pOuts, int * pNumFanouts);
SYCL_EXTERNAL void resynCut(const int *vResynInd, const int *vCutTable,
                            const int *vCutSizes, const int *vNumSaved,
                            const uint64 *htKeys, const uint32 *htValues,
                            int htCapacity, const int *pLevels,
                            uint64 *vSubgTable, int *vSubgLinks, int *vSubgLens,
                            int *pSubgTableNext, unsigned *vTruth,
                            const int *vTruthRanges, const unsigned *vTruthElem,
                            int nMaxCutSize, int nResyn);
SYCL_EXTERNAL void factorFromTruth(const int *vCuts, const int *vCutRanges,
                                   uint64 *vSubgTable, int *vSubgLinks,
                                   int *vSubgLens, int *pSubgTableNext,
                                   const unsigned *vTruth,
                                   const unsigned *vTruthNeg,
                                   const int *vTruthRanges,
                                   const unsigned *vTruthElem, int nResyn);

const int MAX_CUT_SIZE = 15;
const int CUT_TABLE_SIZE = 16;
const int SUBG_TABLE_SIZE = 8;
const int STACK_SIZE = 192;
const int MAX_SUBG_SIZE = 256;
const double HT_LOAD_FACTOR = 0.25;




