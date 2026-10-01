#pragma once
#include <sycl/sycl.hpp>
#include "common.h"

// functions for checking and manipulating truth tables
namespace truthUtil {

inline
int isConst0(const unsigned * pIn, int nVars) {
    int w;
    for (w = dUtils::TruthWordNum(nVars) - 1; w >= 0; w--)
        if (pIn[w])
            return 0;
    return 1;
}

inline
int isConst1(const unsigned * pIn, int nVars) {
    int w;
    for (w = dUtils::TruthWordNum(nVars) - 1; w >= 0; w--)
        if (pIn[w] != ~(unsigned)0)
            return 0;
    return 1;
}

inline
void clear(unsigned * pOut, int nVars) {
    int w;
    for (w = dUtils::TruthWordNum(nVars) - 1; w >= 0; w--)
        pOut[w] = 0;
}

inline
void fill(unsigned * pOut, int nVars) {
    int w;
    for (w = dUtils::TruthWordNum(nVars) - 1; w >= 0; w--)
        pOut[w] = ~(unsigned)0;
}

inline
void truthAnd(unsigned * pOut, const unsigned * pIn0, const unsigned * pIn1, int nVars) {
    int w;
    for (w = dUtils::TruthWordNum(nVars) - 1; w >= 0; w--)
        pOut[w] = pIn0[w] & pIn1[w];
}

inline
void truthOr(unsigned * pOut, const unsigned * pIn0, const unsigned * pIn1, int nVars) {
    int w;
    for (w = dUtils::TruthWordNum(nVars) - 1; w >= 0; w--)
        pOut[w] = pIn0[w] | pIn1[w];
}

inline
void truthNot(unsigned * pOut, unsigned * pIn, int nVars) {
    int w;
    for (w = dUtils::TruthWordNum(nVars) - 1; w >= 0; w--)
        pOut[w] = ~pIn[w];
}

inline
void truthSharp(unsigned * pOut, const unsigned * pIn0, const unsigned * pIn1, int nVars) {
    int w;
    for (w = dUtils::TruthWordNum(nVars) - 1; w >= 0; w--)
        pOut[w] = pIn0[w] & ~pIn1[w];
}

inline
int truthEqual(const unsigned * pIn0, const unsigned * pIn1, int nVars) {
    int w;
    for (w = dUtils::TruthWordNum(nVars) - 1; w >= 0; w--)
        if (pIn0[w] != pIn1[w])
            return 0;
    return 1;
}

inline
int varInSupport(const unsigned * pTruth, int nVars, int iVar) {
    // check whether the truth table depends on iVar
    int nWords = dUtils::TruthWordNum(nVars);
    int i, k, Step;

    assert(iVar < nVars);
    switch (iVar) {
    case 0:
        for (i = 0; i < nWords; i++)
            if ((pTruth[i] & 0x55555555) != ((pTruth[i] & 0xAAAAAAAA) >> 1))
                return 1;
        return 0;
    case 1:
        for (i = 0; i < nWords; i++)
            if ((pTruth[i] & 0x33333333) != ((pTruth[i] & 0xCCCCCCCC) >> 2))
                return 1;
        return 0;
    case 2:
        for (i = 0; i < nWords; i++)
            if ((pTruth[i] & 0x0F0F0F0F) != ((pTruth[i] & 0xF0F0F0F0) >> 4))
                return 1;
        return 0;
    case 3:
        for (i = 0; i < nWords; i++)
            if ((pTruth[i] & 0x00FF00FF) != ((pTruth[i] & 0xFF00FF00) >> 8))
                return 1;
        return 0;
    case 4:
        for (i = 0; i < nWords; i++)
            if ((pTruth[i] & 0x0000FFFF) != ((pTruth[i] & 0xFFFF0000) >> 16))
                return 1;
        return 0;
    default:
        Step = (1 << (iVar - 5));
        for (k = 0; k < nWords; k += 2 * Step) {
            for (i = 0; i < Step; i++)
                if (pTruth[i] != pTruth[Step + i])
                    return 1;
            pTruth += 2 * Step;
        }
        return 0;
    }
}

inline
void cofactor0(unsigned * pTruth, int nVars, int iVar) {
    int nWords = dUtils::TruthWordNum(nVars);
    int i, k, Step;

    assert(iVar < nVars);
    switch (iVar) {
    case 0:
        for (i = 0; i < nWords; i++)
            pTruth[i] = (pTruth[i] & 0x55555555) | ((pTruth[i] & 0x55555555) << 1);
        return;
    case 1:
        for (i = 0; i < nWords; i++)
            pTruth[i] = (pTruth[i] & 0x33333333) | ((pTruth[i] & 0x33333333) << 2);
        return;
    case 2:
        for (i = 0; i < nWords; i++)
            pTruth[i] = (pTruth[i] & 0x0F0F0F0F) | ((pTruth[i] & 0x0F0F0F0F) << 4);
        return;
    case 3:
        for (i = 0; i < nWords; i++)
            pTruth[i] = (pTruth[i] & 0x00FF00FF) | ((pTruth[i] & 0x00FF00FF) << 8);
        return;
    case 4:
        for (i = 0; i < nWords; i++)
            pTruth[i] = (pTruth[i] & 0x0000FFFF) | ((pTruth[i] & 0x0000FFFF) << 16);
        return;
    default:
        Step = (1 << (iVar - 5));
        for (k = 0; k < nWords; k += 2 * Step) {
            for (i = 0; i < Step; i++)
                pTruth[Step + i] = pTruth[i];
            pTruth += 2 * Step;
        }
        return;
    }
}

inline
void cofactor1(unsigned * pTruth, int nVars, int iVar) {
    int nWords = dUtils::TruthWordNum(nVars);
    int i, k, Step;

    assert(iVar < nVars);
    switch (iVar) {
    case 0:
        for (i = 0; i < nWords; i++)
            pTruth[i] = (pTruth[i] & 0xAAAAAAAA) | ((pTruth[i] & 0xAAAAAAAA) >> 1);
        return;
    case 1:
        for (i = 0; i < nWords; i++)
            pTruth[i] = (pTruth[i] & 0xCCCCCCCC) | ((pTruth[i] & 0xCCCCCCCC) >> 2);
        return;
    case 2:
        for (i = 0; i < nWords; i++)
            pTruth[i] = (pTruth[i] & 0xF0F0F0F0) | ((pTruth[i] & 0xF0F0F0F0) >> 4);
        return;
    case 3:
        for (i = 0; i < nWords; i++)
            pTruth[i] = (pTruth[i] & 0xFF00FF00) | ((pTruth[i] & 0xFF00FF00) >> 8);
        return;
    case 4:
        for (i = 0; i < nWords; i++)
            pTruth[i] = (pTruth[i] & 0xFFFF0000) | ((pTruth[i] & 0xFFFF0000) >> 16);
        return;
    default:
        Step = (1 << (iVar - 5));
        for (k = 0; k < nWords; k += 2 * Step) {
            for (i = 0; i < Step; i++)
                pTruth[i] = pTruth[Step + i];
            pTruth += 2 * Step;
        }
        return;
    }
}

inline
void stretch(unsigned * pInOut, int nVarS, int nVarB) {
    int w, i, step, nWords;
    if (nVarS == nVarB)
        return;
    assert(nVarS < nVarB);
    step = dUtils::TruthWordNum(nVarS);
    nWords = dUtils::TruthWordNum(nVarB);
    if (step == nWords)
        return;
    assert(step < nWords);
    for (w = 0; w < nWords; w += step)
        for (i = 0; i < step; i++)
            pInOut[w + i] = pInOut[i];
}

}; // namespace truthUtil
