#pragma once
#include <sycl/sycl.hpp>
#include "sop.hpp"
#include "vectors.hpp"
#include "truth_utils.hpp"

namespace sop {

// *************** Minato ISOP algorithm ***************
//
// SYCL/SPIR-V forbids device-side recursion. The Minato ISOP recursion is a
// ternary post-order tree (compute three cofactor covers, then combine) that
// strictly decreases the variable count each level, so its depth is bounded by
// the number of variables. Below it is rewritten in explicit-stack iterative
// form: a DFS with a per-frame stage machine that preserves the exact
// evaluation order and vecsMem pool allocation order of the recursive version.

// One-word (<= 5 vars) ISOP. Returns the resulting truth table; fills *pcRes.
inline unsigned
minatoIsop5(unsigned uOn_in, unsigned uOnDc_in, int nVars_in, Sop *pcRes_in,
            VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> *vecsMem) {
    const unsigned uMasks[5] = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0,
                                0xFF00FF00, 0xFFFF0000};
    struct Frame {
        unsigned uOn, uOnDc;
        int nVars;
        Sop *pcRes;
        int stage, Var;
        unsigned uOn0, uOn1, uOnDc0, uOnDc1, uRes0, uRes1, uRes2;
        Sop cRes0, cRes1, cRes2;
    };
    Frame st[8]; // depth <= nVars + 1 <= 6
    int sp = 0;
    st[0].uOn = uOn_in; st[0].uOnDc = uOnDc_in; st[0].nVars = nVars_in;
    st[0].pcRes = pcRes_in; st[0].stage = 0;
    unsigned ret = 0;

    while (sp >= 0) {
        Frame &f = st[sp];
        if (f.stage == 0) {
            assert(f.nVars <= 5);
            assert((f.uOn & ~f.uOnDc) == 0);
            if (f.uOn == 0) {
                f.pcRes->nLits = 0; f.pcRes->nCubes = 0; f.pcRes->pCubes = NULL;
                ret = 0; sp--; continue;
            }
            if (f.uOnDc == 0xFFFFFFFF) {
                f.pcRes->nLits = 0; f.pcRes->nCubes = 1;
                f.pcRes->pCubes = vecsMem->fetch(1);
                assert(f.pcRes->pCubes != NULL);
                f.pcRes->pCubes[0] = 0;
                ret = 0xFFFFFFFF; sp--; continue;
            }
            int Var;
            for (Var = f.nVars - 1; Var >= 0; Var--)
                if (truthUtil::varInSupport(&f.uOn, 5, Var) ||
                    truthUtil::varInSupport(&f.uOnDc, 5, Var))
                    break;
            assert(Var >= 0);
            f.Var = Var;
            f.uOn0 = f.uOn1 = f.uOn;
            f.uOnDc0 = f.uOnDc1 = f.uOnDc;
            truthUtil::cofactor0(&f.uOn0, Var + 1, Var);
            truthUtil::cofactor1(&f.uOn1, Var + 1, Var);
            truthUtil::cofactor0(&f.uOnDc0, Var + 1, Var);
            truthUtil::cofactor1(&f.uOnDc1, Var + 1, Var);
            f.stage = 1;
            assert(sp + 1 < 8);
            Frame &c = st[++sp];
            c.uOn = f.uOn0 & ~f.uOnDc1; c.uOnDc = f.uOnDc0;
            c.nVars = f.Var; c.pcRes = &f.cRes0; c.stage = 0;
            continue;
        } else if (f.stage == 1) {
            f.uRes0 = ret;
            f.stage = 2;
            assert(sp + 1 < 8);
            Frame &c = st[++sp];
            c.uOn = f.uOn1 & ~f.uOnDc0; c.uOnDc = f.uOnDc1;
            c.nVars = f.Var; c.pcRes = &f.cRes1; c.stage = 0;
            continue;
        } else if (f.stage == 2) {
            f.uRes1 = ret;
            f.stage = 3;
            assert(sp + 1 < 8);
            Frame &c = st[++sp];
            c.uOn = (f.uOn0 & ~f.uRes0) | (f.uOn1 & ~f.uRes1);
            c.uOnDc = f.uOnDc0 & f.uOnDc1;
            c.nVars = f.Var; c.pcRes = &f.cRes2; c.stage = 0;
            continue;
        } else { // stage 3: combine
            f.uRes2 = ret;
            Sop *pcRes = f.pcRes, *pcRes0 = &f.cRes0, *pcRes1 = &f.cRes1, *pcRes2 = &f.cRes2;
            int Var = f.Var, i, k;
            pcRes->nLits  = pcRes0->nLits + pcRes1->nLits + pcRes2->nLits +
                            pcRes0->nCubes + pcRes1->nCubes;
            pcRes->nCubes = pcRes0->nCubes + pcRes1->nCubes + pcRes2->nCubes;
            pcRes->pCubes = vecsMem->fetch(pcRes->nCubes);
            assert(pcRes->pCubes != NULL);
            k = 0;
            for (i = 0; i < pcRes0->nCubes; i++)
                pcRes->pCubes[k++] = pcRes0->pCubes[i] | (1 << ((Var << 1) + 0));
            for (i = 0; i < pcRes1->nCubes; i++)
                pcRes->pCubes[k++] = pcRes1->pCubes[i] | (1 << ((Var << 1) + 1));
            for (i = 0; i < pcRes2->nCubes; i++)
                pcRes->pCubes[k++] = pcRes2->pCubes[i];
            assert(k == pcRes->nCubes);
            ret = f.uRes2 | ((f.uRes0 & ~uMasks[Var]) | (f.uRes1 & uMasks[Var]));
            sp--;
            continue;
        }
    }
    return ret;
}

// Multi-word (<= 16 vars) ISOP. Returns the resulting truth table; fills *pcRes.
inline unsigned *
minatoIsopR(const unsigned *puOn_in, const unsigned *puOnDc_in, int nVars_in,
            Sop *pcRes_in, VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> *vecsMem) {
    struct Frame {
        const unsigned *puOn, *puOnDc;
        int nVars;
        Sop *pcRes;
        int stage, Var, nWords, nWordsAll;
        unsigned *pTemp, *pTemp0, *pTemp1;
        const unsigned *puOn0, *puOn1, *puOnDc0, *puOnDc1;
        unsigned *puRes0, *puRes1, *puRes2;
        Sop cRes0, cRes1, cRes2;
    };
    Frame st[18]; // recursion entered only for Var >= 5; depth <= nVars <= 16
    int sp = 0;
    st[0].puOn = puOn_in; st[0].puOnDc = puOnDc_in; st[0].nVars = nVars_in;
    st[0].pcRes = pcRes_in; st[0].stage = 0;
    unsigned *ret = nullptr;

    while (sp >= 0) {
        Frame &f = st[sp];
        if (f.stage == 0) {
            f.nWordsAll = dUtils::TruthWordNum(f.nVars);
            f.pTemp = vecsMem->fetch(f.nWordsAll);
            assert(f.pTemp != NULL);
            if (truthUtil::isConst0(f.puOn, f.nVars)) {
                f.pcRes->nLits = 0; f.pcRes->nCubes = 0; f.pcRes->pCubes = NULL;
                truthUtil::clear(f.pTemp, f.nVars);
                ret = f.pTemp; sp--; continue;
            }
            if (truthUtil::isConst1(f.puOnDc, f.nVars)) {
                f.pcRes->nLits = 0; f.pcRes->nCubes = 1;
                f.pcRes->pCubes = vecsMem->fetch(1);
                assert(f.pcRes->pCubes != NULL);
                f.pcRes->pCubes[0] = 0;
                truthUtil::fill(f.pTemp, f.nVars);
                ret = f.pTemp; sp--; continue;
            }
            int Var;
            for (Var = f.nVars - 1; Var >= 0; Var--)
                if (truthUtil::varInSupport(f.puOn, f.nVars, Var) ||
                    truthUtil::varInSupport(f.puOnDc, f.nVars, Var))
                    break;
            assert(Var >= 0);
            f.Var = Var;
            if (Var < 5) {
                unsigned uRes = minatoIsop5(f.puOn[0], f.puOnDc[0], Var + 1, f.pcRes, vecsMem);
                for (int i = 0; i < f.nWordsAll; i++)
                    f.pTemp[i] = uRes;
                ret = f.pTemp; sp--; continue;
            }
            assert(Var >= 5);
            f.nWords = dUtils::TruthWordNum(Var);
            f.puOn0 = f.puOn;      f.puOn1 = f.puOn + f.nWords;
            f.puOnDc0 = f.puOnDc;  f.puOnDc1 = f.puOnDc + f.nWords;
            f.pTemp0 = f.pTemp;    f.pTemp1 = f.pTemp + f.nWords;
            truthUtil::truthSharp(f.pTemp0, f.puOn0, f.puOnDc1, Var); // f'0
            f.stage = 1;
            assert(sp + 1 < 18);
            Frame &c = st[++sp];
            c.puOn = f.pTemp0; c.puOnDc = f.puOnDc0; c.nVars = f.Var;
            c.pcRes = &f.cRes0; c.stage = 0;
            continue;
        } else if (f.stage == 1) {
            f.puRes0 = ret; // g0
            truthUtil::truthSharp(f.pTemp1, f.puOn1, f.puOnDc0, f.Var); // f'1
            f.stage = 2;
            assert(sp + 1 < 18);
            Frame &c = st[++sp];
            c.puOn = f.pTemp1; c.puOnDc = f.puOnDc1; c.nVars = f.Var;
            c.pcRes = &f.cRes1; c.stage = 0;
            continue;
        } else if (f.stage == 2) {
            f.puRes1 = ret; // g1
            truthUtil::truthSharp(f.pTemp0, f.puOn0, f.puRes0, f.Var); // f''0
            truthUtil::truthSharp(f.pTemp1, f.puOn1, f.puRes1, f.Var); // f''1
            truthUtil::truthOr(f.pTemp0, f.pTemp0, f.pTemp1, f.Var);   // f*
            truthUtil::truthAnd(f.pTemp1, f.puOnDc0, f.puOnDc1, f.Var);
            f.stage = 3;
            assert(sp + 1 < 18);
            Frame &c = st[++sp];
            c.puOn = f.pTemp0; c.puOnDc = f.pTemp1; c.nVars = f.Var;
            c.pcRes = &f.cRes2; c.stage = 0;
            continue;
        } else { // stage 3: combine
            f.puRes2 = ret;
            Sop *pcRes = f.pcRes, *pcRes0 = &f.cRes0, *pcRes1 = &f.cRes1, *pcRes2 = &f.cRes2;
            int Var = f.Var, i, k;
            pcRes->nLits  = pcRes0->nLits + pcRes1->nLits + pcRes2->nLits +
                            pcRes0->nCubes + pcRes1->nCubes;
            pcRes->nCubes = pcRes0->nCubes + pcRes1->nCubes + pcRes2->nCubes;
            pcRes->pCubes = vecsMem->fetch(pcRes->nCubes);
            assert(pcRes->pCubes != NULL);
            k = 0;
            for (i = 0; i < pcRes0->nCubes; i++)
                pcRes->pCubes[k++] = pcRes0->pCubes[i] | (1 << ((Var << 1) + 0));
            for (i = 0; i < pcRes1->nCubes; i++)
                pcRes->pCubes[k++] = pcRes1->pCubes[i] | (1 << ((Var << 1) + 1));
            for (i = 0; i < pcRes2->nCubes; i++)
                pcRes->pCubes[k++] = pcRes2->pCubes[i];
            assert(k == pcRes->nCubes);
            // create the resulting truth table
            truthUtil::truthOr(f.pTemp0, f.puRes0, f.puRes2, Var);
            truthUtil::truthOr(f.pTemp1, f.puRes1, f.puRes2, Var);
            int nw = f.nWords << 1;
            for (i = 1; i < f.nWordsAll / nw; i++)
                for (k = 0; k < nw; k++)
                    f.pTemp[i * nw + k] = f.pTemp[k];
            ret = f.pTemp;
            sp--;
            continue;
        }
    }
    return ret;
}

inline
void minatoIsop(const unsigned * puTruth, int nVars, 
                VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem) {
    Sop cRes, * pcRes = &cRes;
    unsigned * pResult, * pTemp;

    vecsMem->shrink(0); // clear the memory

    pResult = minatoIsopR(puTruth, puTruth, nVars, pcRes, vecsMem);
    assert(truthUtil::truthEqual(puTruth, pResult, nVars));

    if (pcRes->nCubes == 0 || (pcRes->nCubes == 1 && pcRes->pCubes[0] == 0)) {
        vecsMem->pArray[0] = 0;
        vecsMem->shrink(pcRes->nCubes);
        return;
    }

    // move the cover representation to the beginning of the memory buffer
    pTemp = vecsMem->fetch(pcRes->nCubes);
    assert(pTemp != NULL);
    for (int i = 0; i < pcRes->nCubes; i++)
        pTemp[i] = pcRes->pCubes[i];
    for (int i = 0; i < pcRes->nCubes; i++)
        vecsMem->pArray[i] = pTemp[i];
    vecsMem->shrink(pcRes->nCubes);
}

} // namespace sop
