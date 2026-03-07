/*
 Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

template <typename T>
class overlay;

template<typename T>
void DetectionOverlayBox(
  sycl::nd_item<2> &item,
  const T*__restrict input,
        T*__restrict output,
  int imgWidth, int imgHeight,
  int x0, int y0, int boxWidth, int boxHeight,
  const float4 color)
{
  const int box_x = item.get_global_id(1);
  const int box_y = item.get_global_id(0);

  if( box_x >= boxWidth || box_y >= boxHeight ) return;

  const int x = box_x + x0;
  const int y = box_y + y0;

  if( x >= imgWidth || y >= imgHeight ) return;

  T px = input[ y * imgWidth + x ];

  const float alpha = color.w() / 255.0f;
  const float ialph = 1.0f - alpha;

  px.x() = alpha * color.x() + ialph * px.x();
  px.y() = alpha * color.y() + ialph * px.y();
  px.z() = alpha * color.z() + ialph * px.z();

  output[y * imgWidth + x] = px;
}

template<typename T>
int DetectionOverlay(
  sycl::queue &q,
  T *input, T *output, uint32_t width, uint32_t height,
  Box *detections, int numDetections, float4 colors)
{
  if( width == 0 || height == 0 || !detections || numDetections == 0)
    return 1;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for( int n=0; n < numDetections; n++ )
  {
    const int boxWidth = detections[n].width;
    const int boxHeight = detections[n].height;
    const int boxLeft = detections[n].left;
    const int boxTop = detections[n].top;

    // launch kernel
    const sycl::range<2> lws (8, 8);
    const sycl::range<2> gws ((boxHeight+7)/8*8, (boxWidth+7)/8*8);

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class overlay<T>>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        DetectionOverlayBox<T>(item, input, output,
          width, height, boxLeft, boxTop, boxWidth, boxHeight, colors);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  return 0;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <width> <height>\n", argv[0]);
    return 1;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int img_size = width * height;
  const int img_size_byte = sizeof(float3) * width * height;

  srand(123);
  float3 *input = (float3*) malloc (img_size_byte);
  float3 *output = (float3*) malloc (img_size_byte);
  float3 *ref_output = (float3*) malloc (img_size_byte);

  for (int i = 0; i < img_size; i++) {
    ref_output[i].x() = input[i].x() = rand() % 256;
    ref_output[i].y() = input[i].y() = rand() % 256;
    ref_output[i].z() = input[i].z() = rand() % 256;
  }

  float3 *d_input = sycl::malloc_device<float3>(img_size, q);
  q.memcpy(d_input, input, img_size_byte);

  float3 *d_output = sycl::malloc_device<float3>(img_size, q);
  q.memcpy(d_output, d_input, img_size_byte);

  const int numDetections = img_size * 0.8f;
  Box* detections = (Box*) malloc (numDetections * sizeof(Box));
  for (int i = 0; i < numDetections; i++) {
    detections[i].width = 64 + rand() % 128;
    detections[i].height = 64 + rand() % 128;
    detections[i].left = rand() % (width - 64);
    detections[i].top = rand() % (height - 64);
  }

  float4 colors = {255, 204, 203, 1};

  DetectionOverlay<float3>(q, d_input, d_output, width, height, detections, numDetections, colors);

  reference<float3>(input, ref_output, width, height, detections, numDetections, colors);

  q.memcpy(output, d_output, img_size_byte).wait();

  bool ok = true;
  for (int i = 0; i < img_size; i++)
    if ((fabsf(ref_output[i].x() - output[i].x()) > 1e-3f) ||
        (fabsf(ref_output[i].y() - output[i].y()) > 1e-3f) ||
        (fabsf(ref_output[i].z() - output[i].z()) > 1e-3f)) {
      printf("Error at index %d\n", i);
      ok = false;
      break;
    }

  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_input, q);
  sycl::free(d_output, q);
  free(input);
  free(output);
  free(ref_output);
  free(detections);
  if (!ok) return 1;
  return 0;
}
