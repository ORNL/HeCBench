// Copyright (c) 2016 Nicolas Weber and Sandra C. Amend / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include "shared.h"

#define THREADS 128
#define WSIZE 32
#define TSIZE (THREADS / WSIZE)

#define TX item.get_local_id(1)
#define PX (item.get_group(1) * TSIZE + (TX / WSIZE))
#define PY item.get_group(0)
#define WTHREAD  (TX % WSIZE)


inline
void normalize(float4& var) {
  var[0] /= var[3];
  var[1] /= var[3];
  var[2] /= var[3];
  var[3] = 1.f;
}

void add(float4& output, const uchar3& color, const float factor) {
  output[0] += (float)color[0] * factor;  
  output[1] += (float)color[1] * factor;  
  output[2] += (float)color[2] * factor;  
  output[3] += factor;
}

inline
float lambda(const Params p, const float dist) {
  if(p.lambda == 0.f)
    return 1.f;
  else if(p.lambda == 1.f)
    return dist;
  return sycl::pow(dist, p.lambda);
}

struct Local {
  float sx, ex, sy, ey;
  uint32_t sxr, syr, exr, eyr, xCount, yCount, pixelCount;

  inline Local(sycl::nd_item<2> &item, const Params& p) {
    sx      = sycl::fmax( PX    * p.pWidth, 0.f);
    ex      = sycl::fmin((PX+1) * p.pWidth, (float)p.iWidth);
    sy      = sycl::fmax( PY    * p.pHeight, 0.f);
    ey      = sycl::fmin((PY+1) * p.pHeight, (float)p.iHeight);

    sxr      = (uint32_t)sycl::floor(sx);
    syr      = (uint32_t)sycl::floor(sy);
    exr      = (uint32_t)sycl::ceil(ex);
    eyr      = (uint32_t)sycl::ceil(ey);
    xCount    = exr - sxr;
    yCount    = eyr - syr;
    pixelCount  = xCount * yCount;
  }
};

inline
float contribution(const Local& l, float f, const uint32_t x, const uint32_t y) {
  if(x < l.sx)    f *= 1.f - (l.sx - x);
  if((x+1.f) > l.ex)  f *= 1.f - ((x+1.f) - l.ex);
  if(y < l.sy)    f *= 1.f - (l.sy - y);
  if((y+1.f) > l.ey)  f *= 1.f - ((y+1.f) - l.ey);
  return f;
}

// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
inline
float4 __shfl_down(sycl::nd_item<2> &item, const float4 var, const uint32_t srcLane)
{
  float4 output;
  auto sg = item.get_sub_group();
  output[0] = sycl::shift_group_left(sg, var[0], srcLane);
  output[1] = sycl::shift_group_left(sg, var[1], srcLane);
  output[2] = sycl::shift_group_left(sg, var[2], srcLane);
  output[3] = sycl::shift_group_left(sg, var[3], srcLane);
  return output;
}

inline
void reduce(sycl::nd_item<2> &item, float4& value) {
  value += __shfl_down(item, value, 16);
  value += __shfl_down(item, value, 8);
  value += __shfl_down(item, value, 4);
  value += __shfl_down(item, value, 2);
  value += __shfl_down(item, value, 1);
}

inline
float distance(const float4& avg, const uchar3& color) {
  const float x = avg[0] - color[0];
  const float y = avg[1] - color[1];
  const float z = avg[2] - color[2];
  return sycl::sqrt(x * x + y * y + z * z) / 441.6729559f; // L2-Norm / sqrt(255^2 * 3)
}

void kernelGuidance(sycl::nd_item<2> &item,
                    const uchar3* __restrict input,
                    uchar3* __restrict patches, const Params p)
{
  if(PX >= p.oWidth || PY >= p.oHeight) return;

  // init
  const Local l(item, p);
  float4 color = (float4)(0);

  // iterate pixels
  for(uint32_t i = WTHREAD; i < l.pixelCount; i += WSIZE) {
    const uint32_t x = l.sxr + (i % l.xCount);
    const uint32_t y = l.syr + (i / l.xCount);

    float f = contribution(l, 1.f, x, y);  

    const uchar3& pixel = input[x + y * p.iWidth];
    color += {pixel[0] * f, pixel[1] * f, pixel[2] * f, f};
  }

  // reduce warps
  reduce(item, color);

  // store results
  if((TX % 32) == 0) {
    normalize(color);
    patches[PX + PY * p.oWidth] = uchar3(color[0], color[1], color[2]);
  }
}

inline
float4 calcAverage(sycl::nd_item<2> &item, const Params& p, 
                   const uchar3* __restrict patches) {
  const float corner = 1.0;
  const float edge   = 2.0;
  const float center = 4.0;

  // calculate average color
  float4 avg = (float4)(0);

  // TOP
  if(PY > 0) {
    if(PX > 0) 
      add(avg, patches[(PX - 1) + (PY - 1) * p.oWidth], corner);

    add(avg, patches[(PX) + (PY - 1) * p.oWidth], edge);

    if((PX+1) < p.oWidth)
      add(avg, patches[(PX + 1) + (PY - 1) * p.oWidth], corner);
  }

  // LEFT
  if(PX > 0) 
    add(avg, patches[(PX - 1) + (PY) * p.oWidth], edge);

  // CENTER
  add(avg, patches[(PX) + (PY) * p.oWidth], center);

  // RIGHT
  if((PX+1) < p.oWidth)
    add(avg, patches[(PX + 1) + (PY) * p.oWidth], edge);

  // BOTTOM
  if((PY+1) < p.oHeight) {
    if(PX > 0) 
      add(avg, patches[(PX - 1) + (PY + 1) * p.oWidth], corner);

    add(avg, patches[(PX) + (PY + 1) * p.oWidth], edge);

    if((PX+1) < p.oWidth)
      add(avg, patches[(PX + 1) + (PY + 1) * p.oWidth], corner);
  }

  normalize(avg);

  return avg;
}

void kernelDownsampling(sycl::nd_item<2> &item,
                        const uchar3* __restrict input,
                        const uchar3* __restrict patches,
                        const Params p,
                              uchar3* __restrict output)
{
  if(PX >= p.oWidth || PY >= p.oHeight) return;

  // init
  const Local l(item, p);
  const float4 avg = calcAverage(item, p, patches);

  float4 color = (float4)(0);

  // iterate pixels
  for(uint32_t i = WTHREAD; i < l.pixelCount; i += WSIZE) {
    const uint32_t x = l.sxr + (i % l.xCount);
    const uint32_t y = l.syr + (i / l.xCount);

    const uchar3& pixel = input[x + y * p.iWidth];
    float f = distance(avg, pixel);

    f = lambda(p, f);
    f = contribution(l, f, x, y);

    add(color, pixel, f);
  }

  // reduce warp
  reduce(item, color);

  if(WTHREAD == 0) {
    uchar3& ref = output[PX + PY * p.oWidth];

    if(color[3] == 0.0f)
      ref = uchar3((unsigned char)avg[0], (unsigned char)avg[1], (unsigned char)avg[2]);
    else {
      normalize(color);
      ref = uchar3((unsigned char)color[0], (unsigned char)color[1], (unsigned char)color[2]);
    }
  }
}

void run(const Params& p, const uchar3* hInput, uchar3* hOutput) {
  const size_t sInput    = p.iWidth * p.iHeight;
  const size_t sOutput  = p.oWidth * p.oHeight;
  const size_t sGuidance  = p.oWidth * p.oHeight;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  uchar3 *dInput = sycl::malloc_device<uchar3>(sInput, q);
  q.memcpy(dInput, hInput, sizeof(uchar3) * sInput);

  uchar3 *dOutput = sycl::malloc_device<uchar3>(sOutput, q);

  uchar3 *dGuidance = sycl::malloc_device<uchar3>(sGuidance, q);

  sycl::range<2> lws (1, THREADS);
  sycl::range<2> gws (p.oHeight, (uint32_t)std::ceil(p.oWidth / (float)TSIZE) * THREADS);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (uint32_t i = 0; i < p.repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class guidance>(sycl::nd_range<2>(gws, lws),
        [=] (sycl::nd_item<2> item) {
        kernelGuidance (item, dInput, dGuidance, p);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class downsample>(sycl::nd_range<2>(gws, lws),
        [=] (sycl::nd_item<2> item) {
        kernelDownsampling (item, dInput, dGuidance, p, dOutput);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / p.repeat);
  
  q.memcpy(hOutput, dOutput, sizeof(uchar3) * sOutput).wait();

  sycl::free(dInput, q);
  sycl::free(dOutput, q);
  sycl::free(dGuidance, q);
}
