/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 */
const static char *const sSDKsample = "HSOpticalFlow";

const float THRESHOLD = 0.05f;

#include "common.h"
#include "flowGold.h"
#include "flowOMP.h"
#include "helper_functions.h"
#include <chrono>
#include <cmath>

using Time = std::chrono::steady_clock;
using float_ms = std::chrono::duration<float, std::chrono::milliseconds::period>;

void WriteFloFile(const char *name, int w, int h, int s, const float *u,
                  const float *v) {
  FILE *stream = fopen(name, "wb");

  if (stream == 0) {
    printf("Could not save flow to \"%s\"\n", name);
    return;
  }

  float data = 202021.25f;
  fwrite(&data, sizeof(float), 1, stream);
  fwrite(&w, sizeof(w), 1, stream);
  fwrite(&h, sizeof(h), 1, stream);

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int pos = j + i * s;
      fwrite(u + pos, sizeof(float), 1, stream);
      fwrite(v + pos, sizeof(float), 1, stream);
    }
  }

  fclose(stream);
}

bool LoadImageAsFP32(float *&img_data, int &img_w, int &img_h, int &img_s,
                     const char *name, const char *exePath) {
  printf("Loading \"%s\" ...\n", name);
  char *name_ = sdkFindFilePath(name, exePath);

  if (!name_) {
    printf("File not found\n");
    return false;
  }

  unsigned char *data = 0;
  unsigned int w = 0, h = 0;
  bool result = sdkLoadPPM4ub(name_, &data, &w, &h);

  if (result == false) {
    printf("Invalid file format\n");
    return false;
  }

  img_w = w;
  img_h = h;
  img_s = iAlignUp(img_w);

  img_data = new float[img_s * h];

  const int widthStep = 4 * img_w;

  for (int i = 0; i < img_h; ++i) {
    for (int j = 0; j < img_w; ++j) {
      img_data[j + i * img_s] = ((float)data[j * 4 + i * widthStep]) / 255.0f;
    }
  }

  return true;
}

bool CompareWithGold(int width, int height, int stride, const float *h_uGold,
                     const float *h_vGold, const float *h_u, const float *h_v) {
  float error = 0.0f;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int pos = j + i * stride;
      error += fabsf(h_u[pos] - h_uGold[pos]) + fabsf(h_v[pos] - h_vGold[pos]);
    }
  }

  error /= (float)(width * height);

  printf("L1 error : %.6f\n", error);

  return (error < THRESHOLD);
}

int main(int argc, char **argv) {
  printf("%s Starting...\n\n", sSDKsample);

  const char *const sourceFrameName = argv[1];
  const char *const targetFrameName = argv[2];

  int width;
  int height;
  int stride;

  float *h_source;
  float *h_target;

  if (!LoadImageAsFP32(h_source, width, height, stride, sourceFrameName,
                       argv[0])) {
    exit(EXIT_FAILURE);
  }

  if (!LoadImageAsFP32(h_target, width, height, stride, targetFrameName,
                       argv[0])) {
    exit(EXIT_FAILURE);
  }

  float *h_uGold = new float[stride * height];
  float *h_vGold = new float[stride * height];
  float *h_u = new float[stride * height];
  float *h_v = new float[stride * height];

  const float alpha = 0.2f;
  const int nLevels = 5;
  const int nSolverIters = 500;
  const int nWarpIters = 3;

  auto start = Time::now();
  ComputeFlowGold(h_source, h_target, width, height, stride, alpha, nLevels,
                  nWarpIters, nSolverIters, h_uGold, h_vGold);
  auto stop = Time::now();

  auto duration = std::chrono::duration_cast<float_ms>(stop - start).count();
  printf("Processing time on CPU: %f (ms)\n", duration);

  start = Time::now();
  ComputeFlowOMP(h_source, h_target, width, height, stride, alpha, nLevels,
                 nWarpIters, nSolverIters, h_u, h_v);
  stop = Time::now();
  duration = std::chrono::duration_cast<float_ms>(stop - start).count();
  printf("Processing time on Device: %f (ms)\n", duration);

  bool status =
      CompareWithGold(width, height, stride, h_uGold, h_vGold, h_u, h_v);

  WriteFloFile("FlowGPU.flo", width, height, stride, h_u, h_v);
  WriteFloFile("FlowCPU.flo", width, height, stride, h_uGold, h_vGold);

  delete[] h_uGold;
  delete[] h_vGold;
  delete[] h_u;
  delete[] h_v;
  delete[] h_source;
  delete[] h_target;

  exit(status ? EXIT_SUCCESS : EXIT_FAILURE);
}
