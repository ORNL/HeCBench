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
#include <omp.h>
#include "reference.h"

template<typename T>
void DetectionOverlayBox(
  const T*__restrict input,
        T*__restrict output,
  int imgWidth, int imgHeight,
  int x0, int y0, int boxWidth, int boxHeight,
  const float4 color) 
{
  #pragma omp target teams distribute parallel for collapse(2) thread_limit(64)
  for(int box_y = 0; box_y < boxHeight; box_y++)
    for(int box_x = 0; box_x < boxWidth; box_x++) {
  
      const int x = box_x + x0;
      const int y = box_y + y0;
      
      if( x < imgWidth && y < imgHeight ) {
      
        T px = input[ y * imgWidth + x ];
        
        const float alpha = color.w / 255.0f;
        const float ialph = 1.0f - alpha;
        
        px.x = alpha * color.x + ialph * px.x;
        px.y = alpha * color.y + ialph * px.y;
        px.z = alpha * color.z + ialph * px.z;
        
        output[y * imgWidth + x] = px;
      }
    }
}

template<typename T>
int DetectionOverlay(
  T* input, T* output, uint32_t width, uint32_t height, 
  Box *detections, int numDetections, float4 colors )
{
  if( !input || !output || width == 0 || height == 0 || !detections || numDetections == 0)
    return 1;
  		
  auto start = std::chrono::steady_clock::now();

  for( int n=0; n < numDetections; n++ )
  {
    const int boxWidth = detections[n].width;
    const int boxHeight = detections[n].height;
    const int boxLeft = detections[n].left;
    const int boxTop = detections[n].top;
    
    // launch kernel
    DetectionOverlayBox<T>(
      input, output, width, height, boxLeft, boxTop, boxWidth, boxHeight, colors);
  }

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

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int img_size = width * height;
  const int img_size_byte = sizeof(float3) * width * height;

  srand(123);
  float3 *input = (float3*) malloc (img_size_byte);
  float3 *output = (float3*) malloc (img_size_byte);
  float3 *ref_output = (float3*) malloc (img_size_byte);

  for (int i = 0; i < img_size; i++) {
    output[i].x = ref_output[i].x = input[i].x = rand() % 256; 
    output[i].y = ref_output[i].y = input[i].y = rand() % 256; 
    output[i].z = ref_output[i].z = input[i].z = rand() % 256; 
  }
   
  float4 colors = {255, 204, 203, 1};

  const int numDetections = img_size * 0.8f;
  Box* detections = (Box*) malloc (numDetections * sizeof(Box));
  for (int i = 0; i < numDetections; i++) {
    detections[i].width = 64 + rand() % 128;
    detections[i].height = 64 + rand() % 128;
    detections[i].left = rand() % (width - 64);
    detections[i].top = rand() % (height - 64);
  }

  // TODO: D2D copy from input to output
  #pragma omp target data map (to: input[0:img_size]) \
                          map (tofrom: output[0:img_size])
  {
    DetectionOverlay<float3>(input, output, width, height, detections, numDetections, colors);  
  }

  reference<float3>(input, ref_output, width, height, detections, numDetections, colors);  

  bool ok = true;
  for (int i = 0; i < img_size; i++) 
    if ((fabsf(ref_output[i].x - output[i].x) > 1e-3f) ||
        (fabsf(ref_output[i].y - output[i].y) > 1e-3f) ||
        (fabsf(ref_output[i].z - output[i].z) > 1e-3f)) {
      printf("Error at index %d\n", i);
      ok = false;
      break;
    }

  printf("%s\n", ok ? "PASS" : "FAIL");

  free(input);
  free(output);
  free(ref_output);
  free(detections);
  if (!ok) return 1;
  return 0;
}
