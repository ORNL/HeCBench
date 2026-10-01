#include <sycl/sycl.hpp>
#include "common.h"
#include "truth.hpp"

/**
 * Get the elementary truth table (i.e., truth table of input variables).
 * Should be launched with only one thread. 
 **/
SYCL_EXTERNAL void Aig::getElemTruthTable(unsigned *vTruthElem, int nVars) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_global_id(2);
    unsigned masks[5] = { 0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000 };
    unsigned * pTruth;

    if (idx == 0) {
        int nWords = dUtils::TruthWordNum(nVars);

        for (int i = 0; i < nVars; i++) {
            pTruth = vTruthElem + i * nWords;
            if (i < 5) {
                for (int k = 0; k < nWords; k++)
                    pTruth[k] = masks[i];
            } else {
                for (int k = 0; k < nWords; k++) {
                    if (k & (1 << (i-5)))
                        pTruth[k] = ~(unsigned)0;
                    else
                        pTruth[k] = 0;
                }
            }
        }
    }
}
