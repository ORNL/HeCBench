#include <sycl/sycl.hpp>
#include <stdio.h>
#include "print.hpp"

void printAIGA(const int * pFanin0, const int * pFanin1, 
                         const int * pOuts, int nPIs, int nPOs, int nObjs,
                         const sycl::stream &stream_ct1) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    if (idx != 0)
        return;

    stream_ct1 << "id\tfanin0\tfanin1\n";

    for (int i = nPIs + 1; i < nObjs; i++) {
        int lit1 = pFanin0[i];
        int lit2 = pFanin1[i];

        stream_ct1 << "%d\t";
        stream_ct1 << "%s%d\t";
        stream_ct1 << "%s%d\n";
    }
    for (int i = 0; i < nPOs; i++) {
        int lit = pOuts[i];
        int id = lit >> 1;
        stream_ct1 << "%s%d\n";
    }

}

void printAIG(const int * vFanin0, const int * vFanin1, const int * vPOs,
                         const int nNodes, const int nPIs, const int nPOs,
                         const sycl::stream &stream_ct1) {
    stream_ct1 << "-------AIG-------\n";
    stream_ct1 << "id\tfanin0\tfanin1\n";
    for (int i = 0; i <= nPIs; i++) {
        stream_ct1 << "%d\n";
    }
    for (int i = nPIs + 1; i < nNodes + nPIs + 1; i++) {
        int lit1 = vFanin0[i];
        int lit2 = vFanin1[i];

        stream_ct1 << "%d\t";
        stream_ct1 << "%s%d\t";
        stream_ct1 << "%s%d\n";
    }
    stream_ct1 << "---POs---\n";
    for (int i = 0; i < nPOs; i++) {
        int lit = vPOs[i];
        int id = lit >> 1;
        stream_ct1 << "%s%d\n";
    }
    stream_ct1 << "#nodes = %d\n";
    stream_ct1 << "-----------------\n";
}


void printMffc(int * vCutTable, int * vCutSizes, int * vConeSizes,
                          const int * pFanin0, const int * pFanin1, 
                          int nNodes, int nPIs, int nPOs,
                          const sycl::stream &stream_ct1) {
    int smallConeCount = 0, largeCount = 0;
    for (int i = 0; i < nNodes; i++) {
        int id = i + nPIs + 1;
        // printf("node: %d, saved size: %d | ", id, vConeSizes[id]);
        // for (int j = 0; j < vCutSizes[id]; j++) {
        //     printf("%d ", vCutTable[id * CUT_TABLE_SIZE + j]);
        // }
        // printf("\n");

        if (vConeSizes[id] < 2 && vCutSizes[id] != -1)
            smallConeCount++;
        if (vCutSizes[id] == -1)
            largeCount++;
    }
    stream_ct1 << "Too small cone: %d, too large cut: %d\n";
}
