#include <sycl/sycl.hpp>
#pragma once

namespace sop {

struct Sop {
    int nLits;
    int nCubes;
    unsigned * pCubes;
};

// note that CUDA can maximally allocate 512 KB (= 2^17 32-bit words) local memory for each thread,
// but we also need to leave some memory spaces for other variables and the function call stack
const int ISOP_FACTOR_MEM_CAP = 1 << 13;


} // namespace sop

// Functions for creating and unbinding subgraph nodes.
// Each subgraph node is stored in a 64-bit variable. 
// The lowest bit records the fanout complement info. 1~31 bit records the second literal, 
// and 32~63 bit records the first literal. The literals are represented in the 2*subg_id(+1 for compl) format. 
// Note, that the fanout complement info is only valid for output node. The fanout status of internal nodes are 
// passed by return values. 
namespace subgUtil {

inline void unbindAndNodeKeyFlag(const uint64 key, int *lit1,
                                          int *lit2, int *pfComp) {
    // make sure highest bit is zero
    *pfComp = (int)(key & 1UL);
    *lit2 = (int)((key & 0xffffffffUL) >> 1);
    *lit1 = (int)(key >> 32);
}

inline uint64 formAndNodeKeyFlag(const int lit1, const int lit2,
                                          const int fComp) {
    // make sure lit1 is smaller than lit2
    assert(lit1 >= 0 && lit2 >= 0 && (fComp == 0 || fComp == 1) && lit1 <= lit2);
    uint32 uLit1 = (uint32)lit1;
    uint32 uLit2 = (uint32)lit2;
    uint64 key = ((uint64)uLit1) << 32 | (uLit2 << 1 | (uint32)fComp);
    return key;
}


template <int nCap>
struct Subg {
    static_assert(nCap <= (1 << 9), "The capacity of Subg should be no larger than 512!\n");

    inline void createConst0() { createSingleExistingVar(1); }
    inline void createConst1() { createSingleExistingVar(0); }

    inline void createSingleExistingVar(int realLit) {
        // create a subgraph equivalent (up to complement) to a existing node
        // realLit is represented using the original node id, rather than subgraph id
        // create a subgraph node that has lit0 = lit1, and fComp = isComp(lit0)

        // the user is responsible for clear the graph first
        pArray[nSize++] = formAndNodeKeyFlag(realLit, realLit, dUtils::AigNodeIsComplement(realLit));
    }

    inline int addNodeAnd(int lit0, int lit1) {
        assert(nSize < nCap);
        if (lit0 > lit1) {
            int temp;
            temp = lit0, lit0 = lit1, lit1 = temp;
        }
        pArray[nSize++] = formAndNodeKeyFlag(lit0, lit1, 0);
        // printf("inserted AND: (%d, %d) at idx=%d\n", lit0, lit1, nSize - 1);
        return dUtils::AigNodeLitCond(nSize - 1, 0);
    }

    inline int addNodeOr(int lit0, int lit1) {
        assert(nSize < nCap);
        if (lit0 > lit1) {
            int temp;
            temp = lit0, lit0 = lit1, lit1 = temp;
        }
        // assert(dUtils::AigNodeNot(lit0) <= dUtils::AigNodeNot(lit1));
        // (device code) unexpected ordering diagnostic removed (printf is
        // variadic and not callable from SYCL device code)
        pArray[nSize++] = formAndNodeKeyFlag(dUtils::AigNodeNot(lit0), dUtils::AigNodeNot(lit1), 1);
        // printf("inserted OR: (%d, %d) at idx=%d\n", lit0, lit1, nSize - 1);
        return dUtils::AigNodeLitCond(nSize - 1, 1);
    }

    inline int isConst() {
        int lit0, lit1, fComp;
        unbindAndNodeKeyFlag(pArray[nSize - 1], &lit0, &lit1, &fComp);
        return dUtils::AigNodeID(lit0) == 0 && lit0 == lit1 && fComp == dUtils::AigNodeIsComplement(lit0);
    }

    int nSize = 0;
    uint64 pArray[nCap];
};

inline int cubeHasLit(unsigned uCube, int i) {
    return (uCube & (unsigned)(1 << i)) > 0;
}
inline unsigned cubeSetLit(unsigned uCube, int i) {
    return uCube | (unsigned)(1 << i);
}
inline unsigned cubeRemLit(unsigned uCube, int i) {
    return uCube & ~(unsigned)(1 << i);
}

inline int cubeContains(unsigned uLarge, unsigned uSmall) {
    return (uLarge & uSmall) == uSmall;
}
inline unsigned cubeSharp(unsigned uCube, unsigned uMask) {
    return uCube & ~uMask;
}

inline int cubeIsMarked(unsigned uCube) {
    return cubeHasLit(uCube, 31);
}
inline unsigned cubeMark(unsigned uCube) {
    return cubeSetLit(uCube, 31);
}
inline unsigned cubeUnmark(unsigned uCube) {
    return cubeRemLit(uCube, 31);
}

}; // namespace subgUtil
