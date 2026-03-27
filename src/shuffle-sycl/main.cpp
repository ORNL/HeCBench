/*
   Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>

#define BUF_SIZE 256
#define PATTERN 0xDEADBEEF


// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input,
                                 unsigned int numGroups, unsigned int subGroupSize) {
  for (unsigned i = 0; i < numGroups; ++i) {
    for (unsigned j = 0; j < subGroupSize; j++) {
      output[i * subGroupSize + j] = input[i * subGroupSize + subGroupSize - j - 1];
    }
  }
}

void verifyBroadcast(const int *out, const int subGroupSize, int pattern = 0)
{
  int expected = pattern;
  if (pattern == 0) {
    for (int i = 0; i < subGroupSize; i++)
      expected += i;
  }
  int errors = 0;
  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != expected) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ERROR @ " << i << ":  " << out[i] << "\n";
      ++errors;
      break;
    }
  }
  if (errors == 0)
    std::cout << "PASS\n";
  else
    std::cout << "FAIL\n";
}

void verifyTransposeMatrix(const float *TransposeMatrix, const float* cpuTransposeMatrix,
            const int total, const int subGroupSize)
{
  int errors = 0;
  float eps = 1.0E-6;
  for (int i = 0; i < total; i++) {
    if (std::fabs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ITEM: " << i <<
        " cpu: " << cpuTransposeMatrix[i] <<
        " gpu: " << TransposeMatrix[i] << "\n";
      errors++;
      break;
    }
  }
  if (errors == 0)
    std::cout << "PASS\n";
  else
    std::cout << "FAIL\n";
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <repeat for broadcast> <repeat for matrix transpose>\n";
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int repeat2 = atoi(argv[2]);

  // CPU support may be unavailable for the shuffle instructions
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  std::cout << "Broadcast using the shuffle xor function (subgroup sizes 8, 16, and 32) \n";
  int *out = (int *)malloc(sizeof(int) * BUF_SIZE);
  int *d_out = sycl::malloc_device<int>(BUF_SIZE, q);

  sycl::range<1> gws (BUF_SIZE);
  sycl::range<1> lws (BUF_SIZE);

  int warmup = repeat;
  std::chrono::steady_clock::time_point begin, end;

  for (int n = 0; n < warmup + repeat; n++) {
    if (n == warmup) {
      q.wait();
      begin = std::chrono::steady_clock::now();
    }
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class bc_shflxor_sg8>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int value = item.get_local_id(0) & 0x7;
        auto sg = item.get_sub_group();
        for (int mask = 1; mask < 0x7; mask *= 2)
          value += sycl::permute_group_by_xor(sg, value, mask);
        d_out[item.get_global_id(0)] = value;
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  std::cout << "Average kernel time (subgroup size = 8): "
            << time * 1e-3f / repeat << "(us)\n";

  q.memcpy(out, d_out, sizeof(int) * BUF_SIZE).wait();

  verifyBroadcast(out, 8);

  //=====================================================================================================
  for (int n = 0; n < warmup + repeat; n++) {
    if (n == warmup) {
      q.wait();
      begin = std::chrono::steady_clock::now();
    }
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class bc_shflxor_sg16>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int value = item.get_local_id(0) & 0xf;
        auto sg = item.get_sub_group();
        for (int mask = 1; mask < 0xf; mask *= 2)
          value += sycl::permute_group_by_xor(sg, value, mask);
        d_out[item.get_global_id(0)] = value;
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  std::cout << "Average kernel time (subgroup size = 16): "
            << time * 1e-3f / repeat << "(us)\n";

  q.memcpy(out, d_out, sizeof(int) * BUF_SIZE).wait();

  verifyBroadcast(out, 16);

  for (int n = 0; n < warmup + repeat; n++) {
    if (n == warmup) {
      q.wait();
      begin = std::chrono::steady_clock::now();
    }
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class bc_shflxor_sg32>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int value = item.get_local_id(0) & 0x1f;
        auto sg = item.get_sub_group();
        for (int mask = 1; mask < 0x1f; mask *= 2)
          value += sycl::permute_group_by_xor(sg, value, mask);
        d_out[item.get_global_id(0)] = value;
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  std::cout << "Average kernel time (subgroup size = 32): "
            << time * 1e-3f / repeat << "(us)\n";

  q.memcpy(out, d_out, sizeof(int) * BUF_SIZE).wait();

  verifyBroadcast(out, 32);
  //=====================================================================================================
  std::cout << "Broadcast using the shuffle function (subgroup sizes 8, 16, and 32) \n";

  for (int n = 0; n < warmup + repeat; n++) {
    if (n == warmup) {
      q.wait();
      begin = std::chrono::steady_clock::now();
    }
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class bc_shfl_sg8>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int value = (item.get_local_id(0) & 0x7) == 0 ? PATTERN : 0;
        int out_v = sycl::select_from_group(item.get_sub_group(), value, 0);
        d_out[item.get_global_id(0)] = out_v;
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  std::cout << "Average kernel time (subgroup size = 8): "
            << time * 1e-3f / repeat << "(us)\n";

  q.memcpy(out, d_out, sizeof(int) * BUF_SIZE).wait();

  verifyBroadcast(out, 8, PATTERN);

  //=====================================================================================================
  for (int n = 0; n < warmup + repeat; n++) {
    if (n == warmup) {
      q.wait();
      begin = std::chrono::steady_clock::now();
    }
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class bc_shfl_sg16>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int value = (item.get_local_id(0) & 0xf) == 0 ? PATTERN : 0;
        int out_v = sycl::select_from_group(item.get_sub_group(), value, 0);
        d_out[item.get_global_id(0)] = out_v;
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  std::cout << "Average kernel time (subgroup size = 16): "
            << time * 1e-3f / repeat << "(us)\n";

  q.memcpy(out, d_out, sizeof(int) * BUF_SIZE).wait();

  verifyBroadcast(out, 16, PATTERN);

  //=====================================================================================================
  for (int n = 0; n < warmup + repeat; n++) {
    if (n == warmup) {
      q.wait();
      begin = std::chrono::steady_clock::now();
    }
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class bc_shfl_sg32>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int value = (item.get_local_id(0) & 0x1f) == 0 ? PATTERN : 0;
        int out_v = sycl::select_from_group(item.get_sub_group(), value, 0);
        d_out[item.get_global_id(0)] = out_v;
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  std::cout << "Average kernel time (subgroup size = 32): "
            << time * 1e-3f / repeat << "(us)\n";

  q.memcpy(out, d_out, sizeof(int) * BUF_SIZE).wait();

  verifyBroadcast(out, 32, PATTERN);

  free(out);
  sycl::free(d_out, q);

  //=====================================================================================================
  std::cout << "matrix transpose using the shuffle function (subgroup sizes are 8, 16, and 32)\n";

  const int total = 1 << 27;  // total number of elements in a matrix

  float* Matrix = (float*)malloc(total * sizeof(float));
  float* TransposeMatrix = (float*)malloc(total * sizeof(float));
  float* cpuTransposeMatrix = (float*)malloc(total * sizeof(float));

  // initialize the input data
  for (int i = 0; i < total; i++) {
    Matrix[i] = (float)i * 10.0f;
  }

  float *d_Matrix = sycl::malloc_device<float>(total, q);
  float *d_TransposeMatrix = sycl::malloc_device<float>(total, q);
  q.memcpy(d_Matrix, Matrix, total * sizeof(float)).wait();

  warmup = repeat2;

  for (int n = 0; n < warmup + repeat2; n++) {
    if (n == warmup) {
      q.wait();
      begin = std::chrono::steady_clock::now();
    }
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class transpose_shfl_sg8>(
        sycl::nd_range<1>(sycl::range<1>(total), sycl::range<1>(8)), [=] (sycl::nd_item<1> item) {
        unsigned b_start = item.get_local_range(0) * item.get_group(0);
        unsigned b_offs = b_start + item.get_local_id(0);
        unsigned s_offs = item.get_local_range(0) - item.get_local_id(0) - 1;
        float val = d_Matrix[b_offs];
        d_TransposeMatrix[b_offs] = sycl::select_from_group(item.get_sub_group(), val, s_offs);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  std::cout << "Average kernel time (subgroup size = 8): "
            << time * 1e-3f / repeat2 << "(us)\n";

  q.memcpy(TransposeMatrix, d_TransposeMatrix, total * sizeof(float)).wait();

  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, total/8, 8);
  verifyTransposeMatrix(TransposeMatrix, cpuTransposeMatrix, total, 8);

  for (int n = 0; n < warmup + repeat2; n++) {
    if (n == warmup) {
      q.wait();
      begin = std::chrono::steady_clock::now();
    }
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class transpose_shfl_sg16>(
        sycl::nd_range<1>(sycl::range<1>(total), sycl::range<1>(16)), [=] (sycl::nd_item<1> item) {
        unsigned b_start = item.get_local_range(0) * item.get_group(0);
        unsigned b_offs = b_start + item.get_local_id(0);
        unsigned s_offs = item.get_local_range(0) - item.get_local_id(0) - 1;
        float val = d_Matrix[b_offs];
        d_TransposeMatrix[b_offs] = sycl::select_from_group(item.get_sub_group(), val, s_offs);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  std::cout << "Average kernel time (subgroup size = 16): "
            << time * 1e-3f / repeat2 << "(us)\n";

  q.memcpy(TransposeMatrix, d_TransposeMatrix, total * sizeof(float)).wait();

  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, total/16, 16);
  verifyTransposeMatrix(TransposeMatrix, cpuTransposeMatrix, total, 16);

  for (int n = 0; n < warmup + repeat2; n++) {
    if (n == warmup) {
      q.wait();
      begin = std::chrono::steady_clock::now();
    }
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class transpose_shfl_sg32>(
        sycl::nd_range<1>(sycl::range<1>(total), sycl::range<1>(32)), [=] (sycl::nd_item<1> item) {
        unsigned b_start = item.get_local_range(0) * item.get_group(0);
        unsigned b_offs = b_start + item.get_local_id(0);
        unsigned s_offs = item.get_local_range(0) - item.get_local_id(0) - 1;
        float val = d_Matrix[b_offs];
        d_TransposeMatrix[b_offs] = sycl::select_from_group(item.get_sub_group(), val, s_offs);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  std::cout << "Average kernel time (subgroup size = 32): "
            << time * 1e-3f / repeat2 << "(us)\n";

  q.memcpy(TransposeMatrix, d_TransposeMatrix, total * sizeof(float)).wait();

  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, total/32, 32);
  verifyTransposeMatrix(TransposeMatrix, cpuTransposeMatrix, total, 32);

  sycl::free(d_Matrix, q);
  sycl::free(d_TransposeMatrix, q);
  free(Matrix);
  free(TransposeMatrix);
  free(cpuTransposeMatrix);

  return 0;
}
