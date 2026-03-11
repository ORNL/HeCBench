/* Copyright (c) 1993-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

// Device functions
inline float sigmoidf(float in) {
  return 1.f / (1.f + sycl::exp(-in));
}

// Fused kernel
void elementwise(
    sycl::nd_item<1> &item,
    int hiddenSize, int miniBatch,
    const float *__restrict tmp_h,
    const float *__restrict tmp_i,
    const float *__restrict bias,
    float *__restrict linearGates,
    float *__restrict h_out,
    float *__restrict i_out,
    const float *__restrict c_in,
    float *__restrict c_out)
{
  int index = item.get_global_id(0);
  int numElements = miniBatch * hiddenSize;

  if (index >= numElements) return;

  int batch = index / hiddenSize;
  int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;

  float g[4];

  for (int i = 0; i < 4; i++) {
    g[i] = tmp_i[i * hiddenSize + gateIndex] + tmp_h[i * hiddenSize + gateIndex];
    g[i] += bias[i * hiddenSize + index % hiddenSize] + bias[(i + 4) * hiddenSize + index % hiddenSize];
    linearGates[gateIndex + i * hiddenSize] = g[i];
  }

  float in_gate     = 1.f / (1.f + sycl::exp(-g[0]));
  float forget_gate = 1.f / (1.f + sycl::exp(-g[1]));
  float in_gate2    = sycl::tanh(g[2]);
  float out_gate    = 1.f / (1.f + sycl::exp(-g[3]));

  float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);

  c_out[index] = val;

  val = out_gate * sycl::tanh(val);

  h_out[index] = val;
  i_out[index] = val;
}

float LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}

void init (sycl::nd_item<1> &item, float* data, int size) {
  int index = item.get_global_id(0);
  if (index >= size) return;
  unsigned int seed = index ^ size;
  data[index] = LCG_random(&seed);
}

void test(sycl::queue &q, int hiddenSize, int miniBatch, int seqLength, int numLayers,
          float *testOutputi, float *testOutputh, float *testOutputc, double &time)
{
  // Input/output data
  int numElements = hiddenSize * miniBatch;

  int hc_size = (seqLength + 1) * (numLayers) * numElements;
  int i_size = (seqLength) * (numLayers + 1) * numElements;
  int bias_size = numLayers * hiddenSize * 8;
  int tmp_h_size = 4 * numLayers * numElements;
  int tmp_i_size = 4 * seqLength * numElements;

  float *h_data = sycl::malloc_device<float>(hc_size, q);
  float *i_data = sycl::malloc_device<float>(i_size, q);
  float *c_data = sycl::malloc_device<float>(hc_size, q);
  float *bias = sycl::malloc_device<float>(bias_size, q);

  // Workspace
  float *tmp_h = sycl::malloc_device<float>(tmp_h_size, q);
  float *tmp_i = sycl::malloc_device<float>(tmp_i_size, q);

  // Activations
  float *linearGates = sycl::malloc_device<float>(4 * seqLength * numLayers * numElements, q);

  // Initialise with random values on a device
  sycl::range<1> lws (256);
  sycl::range<1> gws_hc ((hc_size + 255)/256*256);
  sycl::range<1> gws_b ((bias_size + 255)/256*256);
  sycl::range<1> gws_tmp_h ((tmp_h_size + 255)/256*256);
  sycl::range<1> gws_tmp_i ((tmp_i_size + 255)/256*256);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class init_h_data>(
      sycl::nd_range<1>(gws_tmp_h, lws), [=] (sycl::nd_item<1> item) {
      init(item, tmp_h, tmp_h_size);
    });
  });

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class init_c_data>(
      sycl::nd_range<1>(gws_hc, lws), [=] (sycl::nd_item<1> item) {
      init(item, c_data, hc_size);
    });
  });

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class init_i_data>(
      sycl::nd_range<1>(gws_tmp_i, lws), [=] (sycl::nd_item<1> item) {
      init(item, tmp_i, tmp_i_size);
    });
  });

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class init_bias_data>(
      sycl::nd_range<1>(gws_b, lws), [=] (sycl::nd_item<1> item) {
      init(item, bias, bias_size);
    });
  });

  q.wait();

  int lStart = 0;
  int lEnd = 0;
  int rStart = 0;
  int rEnd = 0;
  int recurBatchSize = 2;

  sycl::range<1> gws_p ((numElements + 255)/256*256);

  double ktime = 0.0;

  while (true) {
    // Many layer "scheduling".
    if (lEnd == 0) {
      lStart = 0;
      lEnd = 1;
      rStart = 0;
    }
    else {
      // Move "up" and "left"
      lStart++;
      lEnd++;

      rStart -= recurBatchSize;

      // Over the top or off the left, reset to layer 0
      if (lEnd > numLayers || rStart < 0) {
        rStart += (lStart + 1) * recurBatchSize;

        lStart = 0;
        lEnd = 1;
      }

      // Off the right, step up
      while (rStart >= seqLength && lEnd <= numLayers) {
        lStart++;
        lEnd++;
        rStart -= recurBatchSize;
      }

      // Over the top or off the left, done!
      if (lEnd > numLayers || rStart < 0) {
        break;
      }
    }

    rEnd = rStart + recurBatchSize;
    if (rEnd > seqLength) rEnd = seqLength;

    auto start = std::chrono::steady_clock::now();

    for (int layer = lStart; layer < lEnd; layer++) {
      for (int i = rStart; i < rEnd; i++)
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for<class pw>(sycl::nd_range<1>(gws_p, lws), [=] (sycl::nd_item<1> item) {
            elementwise
            (item, hiddenSize, miniBatch,
             tmp_h + 4 * layer * numElements,
             tmp_i + 4 * i * numElements,
             bias + 8 * layer * hiddenSize,
             linearGates + 4 * (i * numElements + layer * seqLength * numElements),
             h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
             i_data + i * numElements + (layer + 1) * seqLength * numElements,
             c_data + i * numElements + layer * (seqLength + 1) * numElements,
             c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements);
	      });
        });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    ktime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  time += ktime;
  //printf("Kernel execution time: %f (s)\n", ktime * 1e-9f);

  q.memcpy(testOutputi, i_data + numLayers * seqLength * numElements,
    seqLength * numElements * sizeof(float));
  for (int layer = 0; layer < numLayers; layer++) {
    q.memcpy(testOutputh + layer * numElements,
      h_data + seqLength * numElements + layer * (seqLength + 1) * numElements,
      numElements * sizeof(float));
    q.memcpy(testOutputc + layer * numElements,
      c_data + seqLength * numElements + layer * (seqLength + 1) * numElements,
      numElements * sizeof(float));
  }
  q.wait();

  sycl::free(h_data, q);
  sycl::free(i_data, q);
  sycl::free(c_data, q);

  sycl::free(bias, q);
  sycl::free(tmp_h, q);
  sycl::free(tmp_i, q);
  sycl::free(linearGates, q);
}

int main(int argc, char* argv[]) {
  int seqLength;
  int numLayers;
  int hiddenSize;
  int miniBatch;
  int numRuns;

  if (argc == 6) {
    seqLength = atoi(argv[1]);
    numLayers = atoi(argv[2]);
    hiddenSize = atoi(argv[3]);
    miniBatch = atoi(argv[4]);
    numRuns = atoi(argv[5]);
  }
  else if (argc == 1) {
    printf("Running with default settings\n");
    seqLength = 100;
    numLayers = 4;
    hiddenSize = 512;
    miniBatch = 64;
    numRuns = 1;
  }
  else {
    printf("Usage: %s <seqLength> <numLayers> <hiddenSize> <miniBatch> <repeat>\n", argv[0]);
    return 1;
  }

  printf("seqLength %d, numLayers %d, hiddenSize %d, miniBatch %d\n",
         seqLength, numLayers, hiddenSize, miniBatch);

  int numElements = hiddenSize * miniBatch;
  float *testOutputi = (float*)malloc(numElements * seqLength  * sizeof(float));
  float *testOutputh = (float*)malloc(numElements * numLayers  * sizeof(float));
  float *testOutputc = (float*)malloc(numElements * numLayers  * sizeof(float));
  float *testOutputi_ref = (float*)malloc(numElements * seqLength  * sizeof(float));
  float *testOutputh_ref = (float*)malloc(numElements * numLayers  * sizeof(float));
  float *testOutputc_ref = (float*)malloc(numElements * numLayers  * sizeof(float));

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  double time = 0.0;

  for (int run = 0; run < numRuns; run++) {
    test(q, hiddenSize, miniBatch, seqLength, numLayers,
         testOutputi, testOutputh, testOutputc, time);
    test_ref(hiddenSize, miniBatch, seqLength, numLayers,
             testOutputi_ref, testOutputh_ref, testOutputc_ref);
  }

  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / numRuns);

  int error = 0;
  for (int m = 0; m < miniBatch; m++) {
    for (int j = 0; j < seqLength; j++) {
      for (int i = 0; i < hiddenSize; i++) {
        if (fabsf(testOutputi[j * numElements + m * hiddenSize + i] -
                  testOutputi_ref[j * numElements + m * hiddenSize + i]) > 1e-4f)
          error++;
      }
    }
    for (int j = 0; j < numLayers; j++) {
      for (int i = 0; i < hiddenSize; i++) {
        if (fabsf(testOutputh[j * numElements + m * hiddenSize + i] -
                  testOutputh_ref[j * numElements + m * hiddenSize + i]) > 1e-4f)
          error++;
        if (fabsf(testOutputc[j * numElements + m * hiddenSize + i] -
                  testOutputc_ref[j * numElements + m * hiddenSize + i]) > 1e-4f)
          error++;
      }
    }
  }

  printf("%s\n", (error == 0) ? "PASS" : "FAIL");

  free(testOutputi);
  free(testOutputh);
  free(testOutputc);
  free(testOutputi_ref);
  free(testOutputh_ref);
  free(testOutputc_ref);
  return 0;
}
