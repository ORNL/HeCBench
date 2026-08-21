/*  Copyright (c) 2011-2016, Robert Wang, email: robertwgh (at) gmail.com
  All rights reserved. https://github.com/robertwgh/cuLDPC

  Implementation of LDPC decoding algorithm.

  The details of implementation can be found from the following papers:
  1. Wang, G., Wu, M., Sun, Y., & Cavallaro, J. R. (2011, June). A massively parallel implementation of QC-LDPC decoder on GPU. In Application Specific Processors (SASP), 2011 IEEE 9th Symposium on (pp. 82-85). IEEE.
  2. Wang, G., Wu, M., Yin, B., & Cavallaro, J. R. (2013, December). High throughput low latency LDPC decoding on GPU for SDR systems. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE (pp. 1258-1261). IEEE.

  The current release is close to the GlobalSIP2013 paper.
 */

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "LDPC.h"
#include "matrix.h"
#include "kernel.cpp"

float sigma ;
int *info_bin ;

// The check-node update rules the benchmark exercises, in the order they run.
#define NUM_ALGO 2
static const char *const algo_name[NUM_ALGO] = {"normalized min-sum", "log-SPA"};

// Kernel names, templated so that each check-node rule gets a distinct type.
template <bool> class cnp_kernel;
template <bool> class cnp_kernel2;
template <bool> class vnp_kernel;
template <bool> class vnp_kernel2;

// Decode the codewords currently held in llr_gpu MAX_SIM times and return the
// accumulated kernel time in nanoseconds. The check-node rule is a template
// parameter so that both variants live in one binary at no run-time cost.
template <bool MinSum>
static float run_decode(
    sycl::queue &q,
    float * d_llr,
    float * d_dt,
    float * d_R,
    int * d_hard_decision,
    const char * d_h_element_count1,
    const h_element * d_h_compact1,
    const char * d_h_element_count2,
    const h_element * d_h_compact2,
    const float * llr_gpu,
    int memorySize_llr_gpu)
{
  // Define kernel dimension
  sycl::range<2> gws(MCW * CW, BLK_ROW * BLOCK_SIZE_X); // dim of the thread blocks
  sycl::range<2> lws(CW, BLOCK_SIZE_X);
  int sharedRCacheSize = THREADS_PER_BLOCK * NON_EMPTY_ELMENT;

  sycl::range<2> gws2(MCW * CW, BLK_COL * BLOCK_SIZE_X);
  sycl::range<2> lws2(CW, BLOCK_SIZE_X);

  float total_time = 0.f;

  for(int j = 0; j < MAX_SIM; j++)
  {
    // Transfer LLR data into device.
    q.memcpy(d_llr, llr_gpu, memorySize_llr_gpu).wait();

    // kernel launch
    auto start = std::chrono::steady_clock::now();

    for(int ii = 0; ii < MAX_ITERATION; ii++)
    {

      // run check-node processing kernel
      // TODO: run a special kernel the first iteration?
      if(ii == 0) {
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for<cnp_kernel<MinSum>>(
            sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
            ldpc_cnp_kernel_1st_iter<MinSum> (
              d_llr,
              d_dt,
              d_R,
              d_h_element_count1,
              d_h_compact1,
              item);
          });
        });
      } else {
        q.submit([&] (sycl::handler &cgh) {
          sycl::local_accessor<float, 1> RCache (sycl::range<1>(sharedRCacheSize), cgh);
          cgh.parallel_for<cnp_kernel2<MinSum>>(
            sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
            ldpc_cnp_kernel<MinSum>(
              d_llr,
              d_dt,
              d_R,
              d_h_element_count1,
              d_h_compact1,
              RCache.get_multi_ptr<sycl::access::decorated::no>().get(),
              item);
          });
        });
      }

      // run variable-node processing kernel
      // for the last iteration we run a special
      // kernel. this is because we can make a hard
      // decision instead of writing back the belief
      // for the value of each bit.
      if(ii < MAX_ITERATION - 1) {
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for<vnp_kernel<MinSum>>(
            sycl::nd_range<2>(gws2, lws2), [=] (sycl::nd_item<2> item) {
            ldpc_vnp_kernel_normal(
               d_llr,
               d_dt,
               d_h_element_count2,
               d_h_compact2,
               item);
          });
        });
      } else {
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for<vnp_kernel2<MinSum>>(
            sycl::nd_range<2>(gws2, lws2), [=] (sycl::nd_item<2> item) {
            ldpc_vnp_kernel_last_iter(
               d_llr,
               d_dt,
               d_hard_decision,
               d_h_element_count2,
               d_h_compact2,
               item);
          });
        });
      }
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  } // end of MAX-SIM

  return total_time;
}

int main()
{
  printf("GPU LDPC Decoder\r\nComputing...\r\n");

// For cnp kernel
#if MODE == WIMAX
  const char h_element_count1[BLK_ROW] = {6, 7, 7, 6, 6, 7, 6, 6, 7, 6, 6, 6};
  const char h_element_count2[BLK_COL] = {3, 3, 6, 3, 3, 6, 3, 6, 3, 6, 3, 6, \
                                          3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
#else
  const char h_element_count1[BLK_ROW] = {7, 7, 7, 7, 7, 7, 8, 7, 7, 7, 7, 8};
  const char h_element_count2[BLK_COL] = {11,4, 3, 3,11, 3, 3, 3,11, 3, 3, 3, \
                                          3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
#endif

  h_element h_compact1 [H_COMPACT1_COL*H_COMPACT1_ROW]; // for update dt, R
  h_element h_element_temp;

  // init the compact matrix
  for(int i = 0; i < H_COMPACT1_COL; i++)
  {
    for(int j = 0; j < H_COMPACT1_ROW; j ++)
    {
      h_element_temp.x = 0;
      h_element_temp.y = 0;
      h_element_temp.value = -1;
      h_element_temp.valid = 0;
      h_compact1[i*H_COMPACT1_ROW+j] = h_element_temp; // h[i][0-11], the same column
    }
  }

  // scan the h matrix, and gengerate compact mode of h
  for(int i = 0; i < BLK_ROW; i++)
  {
    int k = 0;
    for(int j = 0; j <  BLK_COL; j ++)
    {
      if(h_base[i][j] != -1)
      {
        h_element_temp.x = i;
        h_element_temp.y = j;
        h_element_temp.value = h_base[i][j];
        h_element_temp.valid = 1;
        h_compact1[k*H_COMPACT1_ROW+i] = h_element_temp; // h[i][0-11], the same column
        k++;
      }
    }
    // printf("row %d, #element=%d\n", i, k);
  }

  // h_compact2
  h_element h_compact2 [H_COMPACT2_ROW*H_COMPACT2_COL]; // for update llr

  // init the compact matrix
  for(int i = 0; i < H_COMPACT2_ROW; i++)
  {
    for(int j = 0; j < H_COMPACT2_COL; j ++)
    {
      h_element_temp.x = 0;
      h_element_temp.y = 0;
      h_element_temp.value = -1;
      h_element_temp.valid = 0;
      h_compact2[i*H_COMPACT2_COL+j] = h_element_temp;
    }
  }

  for(int j = 0; j < BLK_COL; j++)
  {
    int k = 0;
    for(int i = 0; i < BLK_ROW; i ++)
    {
      if(h_base[i][j] != -1)
      {
        // although h is transposed, the (x,y) is still (iBlkRow, iBlkCol)
        h_element_temp.x = i;
        h_element_temp.y = j;
        h_element_temp.value = h_base[i][j];
        h_element_temp.valid = 1;
        h_compact2[k*H_COMPACT2_COL+j] = h_element_temp;
        k++;
      }
    }
  }

  //int memorySize_h_base = BLK_ROW * BLK_COL * sizeof(int);
  int wordSize_h_compact1 = H_COMPACT1_ROW * H_COMPACT1_COL;
  int wordSize_h_compact2 = H_COMPACT2_ROW * H_COMPACT2_COL;

  int memorySize_infobits = INFO_LEN * sizeof(int);
  int memorySize_codeword = CODEWORD_LEN * sizeof(int);
  int memorySize_llr = CODEWORD_LEN * sizeof(float);

  info_bin = (int *) malloc(memorySize_infobits) ;
  int *codeword = (int *) malloc(memorySize_codeword) ;
  float *trans = (float *) malloc(memorySize_llr) ;
  float *recv = (float *) malloc(memorySize_llr) ;
  float *llr = (float *) malloc(memorySize_llr) ;

  float rate = (float)0.5f;

  //////////////////////////////////////////////////////////////////////////////////
  // all the variables Starting with _gpu is used in host code and for gpu computation
  int wordSize_llr = MCW *  CW * CODEWORD_LEN;
  int wordSize_dt = MCW *  CW * ROW * BLK_COL;
  int wordSize_R = MCW *  CW * ROW * BLK_COL;
  int wordSize_hard_decision = MCW * CW * CODEWORD_LEN;

  int memorySize_infobits_gpu = MCW * CW * memorySize_infobits ;
  int memorySize_llr_gpu = wordSize_llr * sizeof(float);
  //int memorySize_dt_gpu = wordSize_dt * sizeof(float);
  //int memorySize_R_gpu = wordSize_R * sizeof(float);
  int memorySize_hard_decision_gpu = wordSize_hard_decision * sizeof(int);

  int *info_bin_gpu;
  float *llr_gpu;
  int * hard_decision_gpu;

  info_bin_gpu = (int *) malloc(memorySize_infobits_gpu);
  hard_decision_gpu = (int *) malloc(memorySize_hard_decision_gpu);
  llr_gpu = (float *) malloc(memorySize_llr_gpu);

  error_result this_error;

  int total_frame_error[NUM_ALGO] = {0};
  int total_bit_error[NUM_ALGO] = {0};
  int total_codeword = 0;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int memorySize_h_compact1 = H_COMPACT1_ROW * H_COMPACT1_COL * sizeof(h_element);
  int memorySize_h_compact2 = H_COMPACT2_ROW * H_COMPACT2_COL * sizeof(h_element);

  float *d_llr = sycl::malloc_device<float>(wordSize_llr, q);
  float *d_dt = sycl::malloc_device<float>(wordSize_dt, q);
  float *d_R = sycl::malloc_device<float>(wordSize_R, q);
  int *d_hard_decision = sycl::malloc_device<int>(wordSize_hard_decision, q);
  h_element *d_h_compact1 = sycl::malloc_device<h_element>(wordSize_h_compact1, q);
  h_element *d_h_compact2 = sycl::malloc_device<h_element>(wordSize_h_compact2, q);
  char *d_h_element_count1 = sycl::malloc_device<char>(BLK_ROW, q);
  char *d_h_element_count2 = sycl::malloc_device<char>(BLK_COL, q);

  q.memcpy(d_h_element_count1, h_element_count1, BLK_ROW);
  q.memcpy(d_h_element_count2, h_element_count2, BLK_COL);
  q.memcpy(d_h_compact1, h_compact1, memorySize_h_compact1);
  q.memcpy(d_h_compact2, h_compact2, memorySize_h_compact2);

  srand(69012);

  for(int snri = 0; snri < NUM_SNR; snri++)
  {
    float snr = snr_array[snri];
    sigma = 1.0f/sqrt(2.0f*rate*pow(10.0f,(snr/10.0f)));

    total_codeword = 0;
    for(int algo = 0; algo < NUM_ALGO; algo++)
    {
      total_frame_error[algo] = 0;
      total_bit_error[algo] = 0;
    }

    // Adjust MIN_CODWORD in LDPC.h to reduce simulation time
    while ( (total_frame_error[0] <= MIN_FER ||
             total_frame_error[1] <= MIN_FER) &&
            (total_codeword <= MIN_CODEWORD))
    {
      total_codeword += CW * MCW;

      for(int i = 0; i < CW * MCW; i++)
      {
        // generate random data
        info_gen (info_bin, rand());

        // encode the data
        structure_encode (info_bin, codeword, h_base);

        // BPSK modulation
        modulation (codeword, trans);

        // additive white Gaussian noise
        awgn (trans, recv, rand());

        // LLR init
        llr_init (llr, recv);

        // copy the info_bin and llr to the total memory
        memcpy(info_bin_gpu + i * INFO_LEN, info_bin, memorySize_infobits);
        memcpy(llr_gpu + i * CODEWORD_LEN, llr, memorySize_llr);
      }

      // Both check-node rules decode the same received block, so their error
      // rates are directly comparable.
      for(int algo = 0; algo < NUM_ALGO; algo++)
      {
        float total_time = (algo == 0)
          ? run_decode<true>(q, d_llr, d_dt, d_R, d_hard_decision,
                             d_h_element_count1, d_h_compact1,
                             d_h_element_count2, d_h_compact2,
                             llr_gpu, memorySize_llr_gpu)
          : run_decode<false>(q, d_llr, d_dt, d_R, d_hard_decision,
                              d_h_element_count1, d_h_compact1,
                              d_h_element_count2, d_h_compact2,
                              llr_gpu, memorySize_llr_gpu);

        // Every repetition inside run_decode decodes the same block of codewords,
        // so the error statistics are collected once instead of MAX_SIM times.
        q.memcpy(hard_decision_gpu, d_hard_decision, memorySize_hard_decision_gpu).wait();

        this_error = error_check(info_bin_gpu, hard_decision_gpu);
        total_bit_error[algo] += this_error.bit_error;
        total_frame_error[algo] += this_error.frame_error;

        printf ("\n");
        printf ("Check-node update: %s\n", algo_name[algo]);
        printf ("Total kernel execution time: %f (s)\n", total_time * 1e-9f);
        printf ("# codewords = %d, CW=%d, MCW=%d\n",total_codeword, CW, MCW);
        printf ("total bit error = %d\n", total_bit_error[algo]);
        printf ("total frame error = %d\n", total_frame_error[algo]);
        printf ("BER = %1.2e, FER = %1.2e\n",
            (float) total_bit_error[algo]/total_codeword/INFO_LEN,
            (float) total_frame_error[algo]/total_codeword);
      }
    } // end of the MAX frame error.
  }// end of the snr loop

  sycl::free(d_llr, q);
  sycl::free(d_dt, q);
  sycl::free(d_R, q);
  sycl::free(d_hard_decision, q);
  sycl::free(d_h_compact1, q);
  sycl::free(d_h_compact2, q);
  sycl::free(d_h_element_count1, q);
  sycl::free(d_h_element_count2, q);

  free(info_bin);
  free(codeword);
  free(trans);
  free(recv);
  free(llr);
  free(llr_gpu);
  free(hard_decision_gpu);
  free(info_bin_gpu);

  return 0;
}
