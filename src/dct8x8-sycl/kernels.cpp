/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <sycl/sycl.hpp>
#include "DCT8x8.h"

inline void DCT8(float *D){
    float X07P = D[0] + D[7];
    float X16P = D[1] + D[6];
    float X25P = D[2] + D[5];
    float X34P = D[3] + D[4];

    float X07M = D[0] - D[7];
    float X61M = D[6] - D[1];
    float X25M = D[2] - D[5];
    float X43M = D[4] - D[3];

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    D[0] = C_norm * (X07P34PP + X16P25PP);
    D[2] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    D[4] = C_norm * (X07P34PP - X16P25PP);
    D[6] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    D[1] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    D[3] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    D[5] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    D[7] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

inline void IDCT8(float *D){
    float Y04P   = D[0] + D[4];
    float Y2b6eP = C_b * D[2] + C_e * D[6];

    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * D[7] + C_a * D[1] + C_c * D[3] + C_d * D[5];
    float Y7a1fM3d5cMP = C_a * D[7] - C_f * D[1] + C_d * D[3] - C_c * D[5];

    float Y04M   = D[0] - D[4];
    float Y2e6bM = C_e * D[2] - C_b * D[6];

    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * D[1] - C_d * D[7] - C_f * D[3] - C_a * D[5];
    float Y1d7cP3a5fMM = C_d * D[1] + C_c * D[7] - C_a * D[3] + C_f * D[5];

    D[0] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    D[7] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    D[4] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    D[3] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    D[1] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    D[5] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    D[2] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    D[6] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

void DCT8x8_kernel(
    sycl::nd_item<2> &item,
    float *d_Dst,
    float *d_Src,
    float *l_Transpose,
    const unsigned int stride,
    const unsigned int imageH,
    const unsigned int imageW
){
    const unsigned int    localX = item.get_local_id(1);
    const unsigned int    localY = BLOCK_SIZE * item.get_local_id(0);
    const unsigned int modLocalX = localX & (BLOCK_SIZE - 1);
    const unsigned int   globalX = item.get_group(1) * BLOCK_X + localX;
    const unsigned int   globalY = item.get_group(0) * BLOCK_Y + localY;

    //Process only full blocks
    const bool process = (globalX - modLocalX + BLOCK_SIZE - 1 < imageW) && (globalY + BLOCK_SIZE - 1 < imageH);

    float *l_V = &l_Transpose[localY * (BLOCK_X+1) + localX];
    float *l_H = &l_Transpose[(localY + modLocalX) * (BLOCK_X+1) + localX - modLocalX];
    d_Src += globalY * stride + globalX;
    d_Dst += globalY * stride + globalX;

    float D[8];
    if (process) {
        for(unsigned int i = 0; i < BLOCK_SIZE; i++)
            l_V[i * (BLOCK_X + 1)] = d_Src[i * stride];
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (process) {
        for(unsigned int i = 0; i < BLOCK_SIZE; i++)
            D[i] = l_H[i];
        DCT8(D);
        for(unsigned int i = 0; i < BLOCK_SIZE; i++)
            l_H[i] = D[i];
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (!process) return;

    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        D[i] = l_V[i * (BLOCK_X + 1)];
    DCT8(D);

    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        d_Dst[i * stride] = D[i];
}

void IDCT8x8_kernel(
    sycl::nd_item<2> &item,
    float *d_Dst,
    float *d_Src,
    float *l_Transpose,
    const unsigned int stride,
    const unsigned int imageH,
    const unsigned int imageW
){
    const unsigned int    localX = item.get_local_id(1);
    const unsigned int    localY = BLOCK_SIZE * item.get_local_id(0);
    const unsigned int modLocalX = localX & (BLOCK_SIZE - 1);
    const unsigned int   globalX = item.get_group(1) * BLOCK_X + localX;
    const unsigned int   globalY = item.get_group(0) * BLOCK_Y + localY;

    //Process only full blocks
    const bool process = (globalX - modLocalX + BLOCK_SIZE - 1 < imageW) && (globalY + BLOCK_SIZE - 1 < imageH);

    float *l_V = &l_Transpose[localY * (BLOCK_X+1) + localX];
    float *l_H = &l_Transpose[(localY + modLocalX) * (BLOCK_X+1) + localX - modLocalX];
    d_Src += globalY * stride + globalX;
    d_Dst += globalY * stride + globalX;

    float D[8];
    if (process) {
        for(unsigned int i = 0; i < BLOCK_SIZE; i++)
            l_V[i * (BLOCK_X + 1)] = d_Src[i * stride];
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (process) {
        for(unsigned int i = 0; i < BLOCK_SIZE; i++)
            D[i] = l_H[i];
        IDCT8(D);
        for(unsigned int i = 0; i < BLOCK_SIZE; i++)
            l_H[i] = D[i];
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (!process) return;

    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        D[i] = l_V[i * (BLOCK_X + 1)];
    IDCT8(D);
    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        d_Dst[i * stride] = D[i];
}

inline unsigned int iDivUp(unsigned int dividend, unsigned int divisor){
    return dividend / divisor + (dividend % divisor != 0);
}

void DCT8x8(
    sycl::queue &q,
    float *d_Dst,
    float *d_Src,
    unsigned int stride,
    unsigned int imageH,
    unsigned int imageW,
    int dir
){
    size_t localWorkSize[2];
    size_t globalWorkSize[2];
    localWorkSize[0] = BLOCK_X;
    localWorkSize[1] = BLOCK_Y / BLOCK_SIZE;
    globalWorkSize[0] = iDivUp(imageW, BLOCK_X) * localWorkSize[0];
    globalWorkSize[1] = iDivUp(imageH, BLOCK_Y) * localWorkSize[1];

    sycl::range<2> gws (globalWorkSize[1], globalWorkSize[0]);
    sycl::range<2> lws (localWorkSize[1], localWorkSize[0]);

    if (dir == DCT_FORWARD)  {
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> l_Transpose(sycl::range<1>(BLOCK_Y * (BLOCK_X+1)), cgh);
        cgh.parallel_for<class dct8x8>(
          sycl::nd_range<2>(sycl::range<2>(gws), sycl::range<2>(lws)), [=] (sycl::nd_item<2> item) {
          DCT8x8_kernel(item, d_Dst, d_Src,
                         l_Transpose.get_multi_ptr<sycl::access::decorated::no>().get(),
                         stride, imageH, imageW);
        });
      });
    }
    else {
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> l_Transpose(sycl::range<1>(BLOCK_Y * (BLOCK_X+1)), cgh);
        cgh.parallel_for<class idct8x8>(
          sycl::nd_range<2>(sycl::range<2>(gws), sycl::range<2>(lws)), [=] (sycl::nd_item<2> item) {
          IDCT8x8_kernel(item, d_Dst, d_Src,
                         l_Transpose.get_multi_ptr<sycl::access::decorated::no>().get(),
                         stride, imageH, imageW);
        });
      });
    }
}
