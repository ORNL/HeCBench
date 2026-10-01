#pragma once

#include <tuple>

std::tuple<int *, int *, int *, int *, int> 
    balancePerformV2(int nObjs, int nPIs, int nPOs, int nNodes, 
                     int * d_pFanin0, int * d_pFanin1, int * d_pOuts, 
                     int * d_pNumFanouts, int sortDecId = 1);
