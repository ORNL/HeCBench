#ifndef FLOW_GOLD_H
#define FLOW_GOLD_H

void ComputeFlowGold(const float *I0, const float *I1, int width, int height,
                     int stride, float alpha, int nLevels, int nWarpIters,
                     int nIters, float *u, float *v);

#endif
