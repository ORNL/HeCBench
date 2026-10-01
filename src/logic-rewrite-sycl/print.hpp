#include <sycl/sycl.hpp>
#pragma once

void printAIG(const int * vFanin0, const int * vFanin1, const int * vPOs,
                         const int nNodes, const int nPIs, const int nPOs,
                         const sycl::stream &stream_ct1);
void printAIGA(const int * pFanin0, const int * pFanin1, 
                         const int * pOuts, int nPIs, int nPOs, int nObjs,
                         const sycl::stream &stream_ct1);
void printMffc(int * vCutTable, int * vCutSizes, int * vConeSizes,
                          const int * pFanin0, const int * pFanin1, 
                          int nNodes, int nPIs, int nPOs,
                          const sycl::stream &stream_ct1);
