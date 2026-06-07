#ifndef FLOW_OMP_H
#define FLOW_OMP_H

void ComputeFlowOMP(const float *I0, const float *I1, int width, int height,
                    int stride, float alpha, int nLevels, int nWarpIters,
                    int nSolverIters, float *u, float *v);

#endif
