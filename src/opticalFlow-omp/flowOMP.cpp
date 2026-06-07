/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 */
#include "common.h"

#include "addKernel.hpp"
#include "derivativesKernel.hpp"
#include "downscaleKernel.hpp"
#include "solverKernel.hpp"
#include "upscaleKernel.hpp"
#include "warpingKernel.hpp"

void ComputeFlowOMP(const float *I0, const float *I1, int width, int height,
                    int stride, float alpha, int nLevels, int nWarpIters,
                    int nSolverIters, float *d_u, float *d_v) {
  printf("Computing optical flow on GPU (OpenMP target)...\n");

  float **pI0 = new float *[nLevels];
  float **pI1 = new float *[nLevels];

  int *pW = new int[nLevels];
  int *pH = new int[nLevels];
  int *pS = new int[nLevels];

  const int pixelCount = stride * height;

  float *d_tmp = new float[pixelCount];
  float *d_du0 = new float[pixelCount];
  float *d_dv0 = new float[pixelCount];
  float *d_du1 = new float[pixelCount];
  float *d_dv1 = new float[pixelCount];
  float *d_Ix = new float[pixelCount];
  float *d_Iy = new float[pixelCount];
  float *d_Iz = new float[pixelCount];
  //float *d_u = new float[pixelCount];
  //float *d_v = new float[pixelCount];
  float *d_nu = new float[pixelCount];
  float *d_nv = new float[pixelCount];

  #pragma omp target enter data map(alloc : d_tmp[0:pixelCount])
  #pragma omp target enter data map(alloc : d_du0[0:pixelCount])
  #pragma omp target enter data map(alloc : d_dv0[0:pixelCount])
  #pragma omp target enter data map(alloc : d_du1[0:pixelCount])
  #pragma omp target enter data map(alloc : d_dv1[0:pixelCount])
  #pragma omp target enter data map(alloc : d_Ix[0:pixelCount])
  #pragma omp target enter data map(alloc : d_Iy[0:pixelCount])
  #pragma omp target enter data map(alloc : d_Iz[0:pixelCount])
  #pragma omp target enter data map(alloc : d_u[0:pixelCount])
  #pragma omp target enter data map(alloc : d_v[0:pixelCount])
  #pragma omp target enter data map(alloc : d_nu[0:pixelCount])
  #pragma omp target enter data map(alloc : d_nv[0:pixelCount])

  int currentLevel = nLevels - 1;

  pI0[currentLevel] = new float [pixelCount];
  pI1[currentLevel] = new float [pixelCount];
  
  memcpy(pI0[currentLevel], I0, pixelCount * sizeof(float));
  memcpy(pI1[currentLevel], I1, pixelCount * sizeof(float));
 
  #pragma omp target enter data map(to : pI0[currentLevel][0:pixelCount])
  #pragma omp target enter data map(to : pI1[currentLevel][0:pixelCount])

  pW[currentLevel] = width;
  pH[currentLevel] = height;
  pS[currentLevel] = stride;

  for (; currentLevel > 0; --currentLevel) {
    int nw = pW[currentLevel] / 2;
    int nh = pH[currentLevel] / 2;
    int ns = iAlignUp(nw);

    pI0[currentLevel-1] = new float [ns * nh];
    pI1[currentLevel-1] = new float [ns * nh];
    #pragma omp target enter data map(alloc : pI0[currentLevel-1][0:ns*nh])
    #pragma omp target enter data map(alloc : pI1[currentLevel-1][0:ns*nh])

    Downscale(pI0[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, pI0[currentLevel-1]);
    Downscale(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, pI1[currentLevel-1]);
    pW[currentLevel - 1] = nw;
    pH[currentLevel - 1] = nh;
    pS[currentLevel - 1] = ns;
  }

  #pragma omp target teams distribute parallel for
  for (int i = 0; i < pixelCount; i++) {
    d_u[i] = 0;
    d_v[i] = 0;
  }

  for (; currentLevel < nLevels; ++currentLevel) {
    for (int warpIter = 0; warpIter < nWarpIters; ++warpIter) {
      #pragma omp target teams distribute parallel for
      for (int i = 0; i < pixelCount; i++) {
        d_du0[i] = 0;
        d_dv0[i] = 0;
        d_du1[i] = 0;
        d_dv1[i] = 0;
      }

      WarpImage(pI1[currentLevel], pW[currentLevel], pH[currentLevel], pS[currentLevel], d_u, d_v, d_tmp);

      ComputeDerivatives(pI0[currentLevel], d_tmp, pW[currentLevel], pH[currentLevel],
                         pS[currentLevel], d_Ix, d_Iy, d_Iz);

      for (int iter = 0; iter < nSolverIters; ++iter) {
        SolveForUpdate(d_du0, d_dv0, d_Ix, d_Iy, d_Iz, pW[currentLevel],
                       pH[currentLevel], pS[currentLevel], alpha, d_du1, d_dv1);
        Swap(d_du0, d_du1);
        Swap(d_dv0, d_dv1);
      }

      Add(d_u, d_du0, pH[currentLevel] * pS[currentLevel], d_u);
      Add(d_v, d_dv0, pH[currentLevel] * pS[currentLevel], d_v);
    }

    if (currentLevel != nLevels - 1) {
      float scaleX = (float)pW[currentLevel + 1] / (float)pW[currentLevel];

      Upscale(d_u, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleX, d_nu);

      float scaleY = (float)pH[currentLevel + 1] / (float)pH[currentLevel];

      Upscale(d_v, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleY, d_nv);

      Swap(d_u, d_nu);
      Swap(d_v, d_nv);
    }
  }

  #pragma omp target update from (d_u[0:pixelCount])
  #pragma omp target update from (d_v[0:pixelCount])

  for (int l = 0; l < nLevels; ++l) {
    #pragma omp target exit data map(delete : pI0[l][0:pS[l]*pH[l]])
    #pragma omp target exit data map(delete : pI1[l][0:pS[l]*pH[l]])
    delete(pI0[l]);
    delete(pI1[l]);
  }

  #pragma omp target exit data map(delete : d_tmp[0:pixelCount])
  #pragma omp target exit data map(delete : d_du0[0:pixelCount])
  #pragma omp target exit data map(delete : d_dv0[0:pixelCount])
  #pragma omp target exit data map(delete : d_du1[0:pixelCount])
  #pragma omp target exit data map(delete : d_dv1[0:pixelCount])
  #pragma omp target exit data map(delete : d_Ix[0:pixelCount])
  #pragma omp target exit data map(delete : d_Iy[0:pixelCount])
  #pragma omp target exit data map(delete : d_Iz[0:pixelCount])
  #pragma omp target exit data map(delete : d_u[0:pixelCount])
  #pragma omp target exit data map(delete : d_v[0:pixelCount])
  #pragma omp target exit data map(delete : d_nu[0:pixelCount])
  #pragma omp target exit data map(delete : d_nv[0:pixelCount])

  delete[] pI0;
  delete[] pI1;
  delete[] pW;
  delete[] pH;
  delete[] pS;

  delete[] d_tmp;
  delete[] d_du0;
  delete[] d_dv0;
  delete[] d_du1;
  delete[] d_dv1;
  delete[] d_Ix;
  delete[] d_Iy;
  delete[] d_Iz;
  //delete[] d_u;
  //delete[] d_v;
  delete[] d_nu;
  delete[] d_nv;
}
