#pragma once
#include <sycl/sycl.hpp>
#include "sop.hpp"
#include "vectors.hpp"
#include "truth_utils.hpp"

namespace sop {

const int SUBG_CAP = MAX_SUBG_SIZE;

// *************** Algebraic Factoring algorithm ***************
//
// SYCL/SPIR-V forbids device-side recursion. The original ABC factoring code is
// recursive, so below it is rewritten in explicit-stack iterative form. Each
// converted routine walks its recursion tree with a private frame stack and a
// per-frame stage (state machine), preserving the exact evaluation order and,
// crucially, the exact vecsMem pool allocation order of the recursive version
// (the pool is a monotonically growing bump allocator, so DFS order keeps every
// previously fetched pointer valid). Stack depth bounds are derived from
// sopFactor's assert(nVars < 16): at most 2*nVars <= 30 literals for the main
// factoring recursion, and logarithmic depth for the balanced trivial trees.

// ---- non-recursive helpers (unchanged from the CUDA version) --------------

inline void
sopCreateInverse(Sop *cResult, unsigned *vInput, int nInputCubes,
                 VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> *vecsMem) {
    unsigned uCube, uMask = 0;
    cResult->nCubes = 0;
    cResult->pCubes = vecsMem->fetch(nInputCubes);
    for (int i = 0; i < nInputCubes; i++){
        uCube = vInput[i];
        uMask = ((uCube | (uCube >> 1)) & 0x55555555);
        uMask |= (uMask << 1);
        cResult->pCubes[cResult->nCubes++] = uCube ^ uMask;
    }
}

inline unsigned sopCommonCube(Sop *cSop) {
    unsigned uMask;
    int i;
    uMask = ~(unsigned)0;
    for (i = 0; i < cSop->nCubes; i++)
        uMask &= cSop->pCubes[i];
    return uMask;
}

inline void sopMakeCubeFree(Sop *cSop) {
    unsigned uMask;
    int i;
    uMask = sopCommonCube(cSop);
    if ( uMask == 0 )
        return;
    for (i = 0; i < cSop->nCubes; i++)
        cSop->pCubes[i] = subgUtil::cubeSharp(cSop->pCubes[i], uMask);
}

inline void sopDivideByLiteralQuo(Sop *cSop, int iLit) {
    int i, k = 0;
    for (i = 0; i < cSop->nCubes; i++)
        if (subgUtil::cubeHasLit(cSop->pCubes[i], iLit))
            cSop->pCubes[k++] = subgUtil::cubeRemLit(cSop->pCubes[i], iLit);
    cSop->nCubes = k;
}

inline int sopWorstLiteral(Sop *cSop, int nLits) {
    int i, k, iMin, nLitsMin, nLitsCur;
    int fUseFirst = 1;
    iMin = -1;
    nLitsMin = 1000000;
    for (i = 0; i < nLits; i++) {
        nLitsCur = 0;
        for (k = 0; k < cSop->nCubes; k++)
            if (subgUtil::cubeHasLit(cSop->pCubes[k], i))
                nLitsCur++;
        if (nLitsCur < 2)
            continue;
        if (fUseFirst) {
            if (nLitsMin > nLitsCur) {
                nLitsMin = nLitsCur;
                iMin = i;
            }
        }
        else {
            if (nLitsMin >= nLitsCur) {
                nLitsMin = nLitsCur;
                iMin = i;
            }
        }
    }
    if (nLitsMin < 1000000)
        return iMin;
    return -1;
}

// Tail recursion -> loop.
inline void sopDivisorZeroKernelRec(Sop *cSop, int nLits) {
    while (true) {
        int iLit = sopWorstLiteral(cSop, nLits);
        if ( iLit == -1 )
            return;
        sopDivideByLiteralQuo(cSop, iLit);
        sopMakeCubeFree(cSop);
    }
}

inline void sopDup(Sop *cResult, Sop *cSop,
                   VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> *vecsMem) {
    int i;
    cResult->nCubes = 0;
    cResult->pCubes = vecsMem->fetch(cSop->nCubes);
    for (i = 0; i < cSop->nCubes; i++)
        cResult->pCubes[cResult->nCubes++] = cSop->pCubes[i];
}

inline int sopAnyLiteral(Sop *cSop, int nLits) {
    int i, k, nLitsCur;
    for (i = 0; i < nLits; i++) {
        nLitsCur = 0;
        for (k = 0; k < cSop->nCubes; k++)
            if (subgUtil::cubeHasLit(cSop->pCubes[k], i))
                nLitsCur++;
        if (nLitsCur > 1)
            return i;
    }
    return -1;
}

inline int
sopDivisor(Sop *cResult, Sop *cSop, int nLits,
           VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> *vecsMem) {
    if (cSop->nCubes <= 1)
        return 0;
    if (sopAnyLiteral(cSop, nLits) == -1)
        return 0;
    sopDup(cResult, cSop, vecsMem);
    sopDivisorZeroKernelRec(cResult, nLits);
    assert(cResult->nCubes > 0);
    return 1;
}

inline void
sopDivideByCube(Sop *cSop, Sop *cDiv, Sop *vQuo, Sop *vRem,
                VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> *vecsMem) {
    unsigned uCube, uDiv;
    int i;
    assert(cDiv->nCubes == 1);
    uDiv = cDiv->pCubes[0];
    vQuo->nCubes = 0;
    vQuo->pCubes = vecsMem->fetch(cSop->nCubes);
    vRem->nCubes = 0;
    vRem->pCubes = vecsMem->fetch(cSop->nCubes);
    for (i = 0; i < cSop->nCubes; i++) {
        uCube = cSop->pCubes[i];
        if (subgUtil::cubeContains(uCube, uDiv))
            vQuo->pCubes[vQuo->nCubes++] = subgUtil::cubeSharp(uCube, uDiv);
        else
            vRem->pCubes[vRem->nCubes++] = uCube;
    }
}

inline void
sopDivideInternal(Sop *cSop, Sop *cDiv, Sop *vQuo, Sop *vRem,
                  VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> *vecsMem) {
    unsigned uCube, uDiv;
    unsigned uCube2 = 0;
    unsigned uDiv2, uQuo;
    int i, i2, k, k2, nCubesRem;
    assert(cSop->nCubes >= cDiv->nCubes);
    if (cDiv->nCubes == 1) {
        sopDivideByCube(cSop, cDiv, vQuo, vRem, vecsMem);
        return;
    }
    vQuo->nCubes = 0;
    vQuo->pCubes = vecsMem->fetch(cSop->nCubes / cDiv->nCubes);
    for (i = 0; i < cSop->nCubes; i++) {
        uCube = cSop->pCubes[i];
        if (subgUtil::cubeIsMarked(uCube))
            continue;
        uDiv = ~0;
        for (k = 0; k < cDiv->nCubes; k++) {
            uDiv = cDiv->pCubes[k];
            if (subgUtil::cubeContains(uCube, uDiv))
                break;
        }
        if (k == cDiv->nCubes)
            continue;
        uQuo = subgUtil::cubeSharp(uCube, uDiv);
        uDiv2 = ~0;
        for (k2 = 0; k2 < cDiv->nCubes; k2++) {
            uDiv2 = cDiv->pCubes[k2];
            if (k2 == k) continue;
            for (i2 = 0; i2 < cSop->nCubes; i2++) {
                uCube2 = cSop->pCubes[i2];
                if (subgUtil::cubeIsMarked(uCube2))
                    continue;
                if (subgUtil::cubeContains(uCube2, uDiv2) && uQuo == subgUtil::cubeSharp(uCube2, uDiv2))
                    break;
            }
            if (i2 == cSop->nCubes)
                break;
        }
        if (k2 != cDiv->nCubes)
            continue;
        vQuo->pCubes[vQuo->nCubes++] = uQuo;
        cSop->pCubes[i] = subgUtil::cubeMark(uCube);
        for (k2 = 0; k2 < cDiv->nCubes; k2++) {
            uDiv2 = cDiv->pCubes[k2];
            if (k2 == k) continue;
            for (i2 = 0; i2 < cSop->nCubes; i2++) {
                uCube2 = cSop->pCubes[i2];
                if (subgUtil::cubeIsMarked(uCube2))
                    continue;
                if (subgUtil::cubeContains(uCube2, uDiv2) && uQuo == subgUtil::cubeSharp(uCube2, uDiv2))
                    break;
            }
            assert(i2 < cSop->nCubes);
            cSop->pCubes[i2] = subgUtil::cubeMark(uCube2);
        }
    }
    nCubesRem = cSop->nCubes - vQuo->nCubes * cDiv->nCubes;
    vRem->nCubes = 0;
    vRem->pCubes = vecsMem->fetch(nCubesRem);
    for (i = 0; i < cSop->nCubes; i++) {
        uCube = cSop->pCubes[i];
        if (!subgUtil::cubeIsMarked(uCube)) {
            vRem->pCubes[vRem->nCubes++] = uCube;
            continue;
        }
        cSop->pCubes[i] = subgUtil::cubeUnmark(uCube);
    }
    assert(nCubesRem == vRem->nCubes);
}

inline int sopBestLiteral(Sop *cSop, int nLits, unsigned uMask) {
    int i, k, iMax, nLitsMax, nLitsCur;
    int fUseFirst = 1;
    iMax = -1;
    nLitsMax = -1;
    for (i = 0; i < nLits; i++) {
        if (!subgUtil::cubeHasLit(uMask, i))
            continue;
        nLitsCur = 0;
        for (k = 0; k < cSop->nCubes; k++)
            if (subgUtil::cubeHasLit(cSop->pCubes[k], i))
                nLitsCur++;
        if (nLitsCur < 2)
            continue;
        if (fUseFirst) {
            if (nLitsMax < nLitsCur) {
                nLitsMax = nLitsCur;
                iMax = i;
            }
        } else {
            if (nLitsMax <= nLitsCur) {
                nLitsMax = nLitsCur;
                iMax = i;
            }
        }
    }
    if (nLitsMax >= 0)
        return iMax;
    return -1;
}

inline void
sopBestLiteralCover(Sop *cResult, Sop *cSop, unsigned uCube, int nLits,
                    VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> *vecsMem) {
    int iLitBest;
    iLitBest = sopBestLiteral(cSop, nLits, uCube);
    cResult->nCubes = 0;
    cResult->pCubes = vecsMem->fetch(1);
    cResult->pCubes[cResult->nCubes++] = subgUtil::cubeSetLit(0, iLitBest);
}

inline void
sopCommonCubeCover(Sop *cResult, Sop *cSop,
                   VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> *vecsMem) {
    assert(cSop->nCubes > 0);
    cResult->nCubes = 0;
    cResult->pCubes = vecsMem->fetch(1);
    cResult->pCubes[cResult->nCubes++] = sopCommonCube(cSop);
}

// ---- iterative trivial-tree constructors ----------------------------------

// Balanced AND tree over the literals of a single cube in interval [nStart,nFinish).
inline int sopFactorTrivialCube(unsigned uCube, int nStart, int nFinish,
                                subgUtil::Subg<SUBG_CAP> *subg) {
    struct FrameC { int nStart, nFinish, stage, splitI, e1; };
    FrameC st[24];
    int sp = 0;
    st[0] = {nStart, nFinish, 0, 0, 0};
    int ret = 0;
    while (sp >= 0) {
        FrameC &f = st[sp];
        if (f.stage == 0) {
            int i, iLit = -1, nLits = 0;
            for (i = f.nStart; i < f.nFinish; i++)
                if (subgUtil::cubeHasLit(uCube, i)) { iLit = i; nLits++; }
            assert(iLit != -1);
            if (nLits == 1) { ret = iLit; sp--; continue; }
            int nLits1 = nLits / 2;
            nLits = 0;
            for (i = f.nStart; i < f.nFinish; i++)
                if (subgUtil::cubeHasLit(uCube, i)) {
                    if (nLits == nLits1) break;
                    nLits++;
                }
            f.splitI = i;
            f.stage = 1;
            assert(sp + 1 < 24);
            st[++sp] = {f.nStart, f.splitI, 0, 0, 0};
            continue;
        } else if (f.stage == 1) {
            f.e1 = ret;
            f.stage = 2;
            assert(sp + 1 < 24);
            st[++sp] = {f.splitI, f.nFinish, 0, 0, 0};
            continue;
        } else {
            ret = subg->addNodeAnd(f.e1, ret);
            sp--;
            continue;
        }
    }
    return ret;
}

// Balanced OR tree over cubes; leaves are AND trees of single cubes.
inline int sopFactorTrivial(unsigned *pCubes, int nCubes, int nLits,
                            subgUtil::Subg<SUBG_CAP> *subg) {
    struct FrameT { unsigned *pCubes; int nCubes, stage, e1; };
    FrameT st[40];
    int sp = 0;
    st[0] = {pCubes, nCubes, 0, 0};
    int ret = 0;
    while (sp >= 0) {
        FrameT &f = st[sp];
        if (f.stage == 0) {
            if (f.nCubes == 1) {
                ret = sopFactorTrivialCube(f.pCubes[0], 0, nLits, subg);
                sp--; continue;
            }
            int nCubes1 = f.nCubes / 2;
            f.stage = 1;
            assert(sp + 1 < 40);
            st[++sp] = {f.pCubes, nCubes1, 0, 0};
            continue;
        } else if (f.stage == 1) {
            f.e1 = ret;
            int nCubes1 = f.nCubes / 2;
            f.stage = 2;
            assert(sp + 1 < 40);
            st[++sp] = {f.pCubes + nCubes1, f.nCubes - nCubes1, 0, 0};
            continue;
        } else {
            ret = subg->addNodeOr(f.e1, ret);
            sp--;
            continue;
        }
    }
    return ret;
}

// ---- iterative algebraic factoring (sopFactorRec / sopFactorLFRec) --------
//
// Mutually-recursive REC and LF frames are handled by one stack with a `func`
// tag. nVars < 16 => <= 30 literals => recursion depth < 32.
constexpr int FACTOR_STACK = 34;

inline int sopFactorIter(Sop *cSopRoot, int nLits,
                         VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> *vecsMem,
                         subgUtil::Subg<SUBG_CAP> *subg) {
    enum { FN_REC = 0, FN_LF = 1 };
    struct FrameF {
        int func;
        Sop *cSop;     // borrowed input cover
        Sop *cSimple;  // LF: borrowed simple cover
        int stage;
        Sop Div, Quo, Rem, Com;
        int eNodeDiv, eNodeAnd;
    };
    FrameF st[FACTOR_STACK];
    int sp = 0;
    st[0].func = FN_REC;
    st[0].cSop = cSopRoot;
    st[0].cSimple = nullptr;
    st[0].stage = 0;
    int ret = 0;

    while (sp >= 0) {
        FrameF &f = st[sp];
        if (f.func == FN_REC) {
            switch (f.stage) {
            case 0: {
                assert(f.cSop->nCubes > 0);
                if (!sopDivisor(&f.Div, f.cSop, nLits, vecsMem)) {
                    ret = sopFactorTrivial(f.cSop->pCubes, f.cSop->nCubes, nLits, subg);
                    sp--; break;
                }
                sopDivideInternal(f.cSop, &f.Div, &f.Quo, &f.Rem, vecsMem);
                assert(f.Quo.nCubes > 0);
                if (f.Quo.nCubes == 1) {
                    // tail: LF(cSop, &Quo)
                    f.stage = 6;
                    assert(sp + 1 < FACTOR_STACK);
                    FrameF &c = st[++sp];
                    c.func = FN_LF; c.cSop = f.cSop; c.cSimple = &f.Quo; c.stage = 0;
                    break;
                }
                sopMakeCubeFree(&f.Quo);
                sopDivideInternal(f.cSop, &f.Quo, &f.Div, &f.Rem, vecsMem);
                if (sopCommonCube(&f.Div) == 0) {
                    f.stage = 1;
                    assert(sp + 1 < FACTOR_STACK);
                    FrameF &c = st[++sp];
                    c.func = FN_REC; c.cSop = &f.Div; c.cSimple = nullptr; c.stage = 0;
                } else {
                    sopCommonCubeCover(&f.Com, &f.Div, vecsMem);
                    f.stage = 6; // tail: LF(cSop, &Com)
                    assert(sp + 1 < FACTOR_STACK);
                    FrameF &c = st[++sp];
                    c.func = FN_LF; c.cSop = f.cSop; c.cSimple = &f.Com; c.stage = 0;
                }
                break;
            }
            case 1: {
                f.eNodeDiv = ret;
                f.stage = 2;
                assert(sp + 1 < FACTOR_STACK);
                FrameF &c = st[++sp];
                c.func = FN_REC; c.cSop = &f.Quo; c.cSimple = nullptr; c.stage = 0;
                break;
            }
            case 2: {
                int eNodeQuo = ret;
                f.eNodeAnd = subg->addNodeAnd(f.eNodeDiv, eNodeQuo);
                if (f.Rem.nCubes == 0) { ret = f.eNodeAnd; sp--; break; }
                f.stage = 3;
                assert(sp + 1 < FACTOR_STACK);
                FrameF &c = st[++sp];
                c.func = FN_REC; c.cSop = &f.Rem; c.cSimple = nullptr; c.stage = 0;
                break;
            }
            case 3: {
                int eNodeRem = ret;
                ret = subg->addNodeOr(f.eNodeAnd, eNodeRem);
                sp--; break;
            }
            default: // case 6: propagate child (tail LF) result
                sp--; break;
            }
        } else { // FN_LF
            switch (f.stage) {
            case 0: {
                assert(f.cSimple->nCubes == 1);
                sopBestLiteralCover(&f.Div, f.cSop, f.cSimple->pCubes[0], nLits, vecsMem);
                sopDivideByCube(f.cSop, &f.Div, &f.Quo, &f.Rem, vecsMem);
                f.eNodeDiv = sopFactorTrivialCube(f.Div.pCubes[0], 0, nLits, subg);
                f.stage = 1;
                assert(sp + 1 < FACTOR_STACK);
                FrameF &c = st[++sp];
                c.func = FN_REC; c.cSop = &f.Quo; c.cSimple = nullptr; c.stage = 0;
                break;
            }
            case 1: {
                int eNodeQuo = ret;
                f.eNodeAnd = subg->addNodeAnd(f.eNodeDiv, eNodeQuo);
                if (f.Rem.nCubes == 0) { ret = f.eNodeAnd; sp--; break; }
                f.stage = 2;
                assert(sp + 1 < FACTOR_STACK);
                FrameF &c = st[++sp];
                c.func = FN_REC; c.cSop = &f.Rem; c.cSimple = nullptr; c.stage = 0;
                break;
            }
            default: { // case 2
                int eNodeRem = ret;
                ret = subg->addNodeOr(f.eNodeAnd, eNodeRem);
                sp--; break;
            }
            }
        }
    }
    return ret;
}

inline void sopFactor(unsigned *vCover, int nCoverSize, int fCompl,
                      const int *vCuts, int nVars,
                      VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> *vecsMem,
                      subgUtil::Subg<SUBG_CAP> *subg) {
    Sop sop, * cSop = &sop;
    int eRoot;
    assert(nVars < 16);

    // clear subgraph and assign leaves
    subg->nSize = nVars;

    // check for trivial functions
    if (nCoverSize == 0) {
        if (fCompl)
            subg->createConst1();
        else
            subg->createConst0();
        return;
    }
    if (nCoverSize == 1 && vCover[0] == 0) {
        if (fCompl)
            subg->createConst0();
        else
            subg->createConst1();
        return;
    }
    // perform CST
    sopCreateInverse(cSop, vCover, nCoverSize, vecsMem);
    // factor the cover
    eRoot = sopFactorIter(cSop, 2 * nVars, vecsMem, subg);

    // if eRoot is a leaf, then this is the case of the resyned cut is a const or a single var of cut nodes
    if (dUtils::AigNodeID(eRoot) < nVars) {
        assert(subg->nSize == nVars);
        subg->createSingleExistingVar(
            dUtils::AigNodeLitCond(vCuts[dUtils::AigNodeID(eRoot)],
                                   dUtils::AigNodeIsComplement(eRoot) != fCompl)
        );
        return;
    }

    // the complementation info of eRoot is already in the root node
    if (fCompl) {
        uint64 rootNode = subg->pArray[subg->nSize - 1];
        int lit0, lit1, fCompRoot;
        subgUtil::unbindAndNodeKeyFlag(rootNode, &lit0, &lit1, &fCompRoot);
        subg->pArray[subg->nSize - 1] = subgUtil::formAndNodeKeyFlag(lit0, lit1, 1 - fCompRoot);
    }
}

} // namespace sop
