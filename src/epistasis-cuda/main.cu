#include <math.h>
#include <float.h>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include "reference.h"
#include <cstdlib>

using namespace std::chrono;
typedef high_resolution_clock myclock;
typedef duration<float> myduration;

#define MAX_WG_SIZE 256

template <typename T>
T* mem_alloc (const int align, const size_t size) {
  return (T*) aligned_alloc(align, size * sizeof(T));
}

template <typename T>
void mem_free (T* p) {
  free(p);
}

__device__
float gammafunction(unsigned int n)
{   
  if(n == 0)
    return 0.0f;
  float x = ((float)n + 0.5f) * logf((float) n) - ((float)n - 1.0f);
  return x;
}

__global__ void epi(const unsigned int* dev_data_zeros, 
                    const unsigned int* dev_data_ones, 
                    float* dev_scores, 
                    const int num_snp, 
                    const int PP_zeros, 
                    const int PP_ones,
                    const int mask_zeros, 
                    const int mask_ones) 
{
  int i, j, tid, p, k;

  j = blockDim.x * blockIdx.x + threadIdx.x; 
  i = blockDim.y * blockIdx.y + threadIdx.y;
  tid = i * num_snp + j;

  if (j > i && i < num_snp && j < num_snp) {
    unsigned int ft[2 * 9];
    for(k = 0; k < 2 * 9; k++) ft[k] = 0;

    unsigned int t00, t01, t02, t10, t11, t12, t20, t21, t22;
    unsigned int di2, dj2;
    unsigned int* SNPi;
    unsigned int* SNPj;

    // Phenotype 0
    SNPi = (unsigned int*) &dev_data_zeros[i * 2];
    SNPj = (unsigned int*) &dev_data_zeros[j * 2];

    #pragma unroll 1
    for (p = 0; p < 2 * PP_zeros * num_snp - 2 * num_snp; p += 2 * num_snp) {
      di2 = ~(SNPi[p] | SNPi[p + 1]);
      dj2 = ~(SNPj[p] | SNPj[p + 1]);

      t00 = SNPi[p] & SNPj[p];
      t01 = SNPi[p] & SNPj[p + 1];
      t02 = SNPi[p] & dj2;
      t10 = SNPi[p + 1] & SNPj[p];
      t11 = SNPi[p + 1] & SNPj[p + 1];
      t12 = SNPi[p + 1] & dj2;
      t20 = di2 & SNPj[p];
      t21 = di2 & SNPj[p + 1];
      t22 = di2 & dj2;

      ft[0] += __popc(t00);
      ft[1] += __popc(t01);
      ft[2] += __popc(t02);
      ft[3] += __popc(t10);
      ft[4] += __popc(t11);
      ft[5] += __popc(t12);
      ft[6] += __popc(t20);
      ft[7] += __popc(t21);
      ft[8] += __popc(t22);
    }

    // remainder
    p = 2 * PP_zeros * num_snp - 2 * num_snp;
    di2 = ~(SNPi[p] | SNPi[p + 1]);
    dj2 = ~(SNPj[p] | SNPj[p + 1]);
    di2 = di2 & mask_zeros;
    dj2 = dj2 & mask_zeros;

    t00 = SNPi[p] & SNPj[p];
    t01 = SNPi[p] & SNPj[p + 1];
    t02 = SNPi[p] & dj2;
    t10 = SNPi[p + 1] & SNPj[p];
    t11 = SNPi[p + 1] & SNPj[p + 1];
    t12 = SNPi[p + 1] & dj2;
    t20 = di2 & SNPj[p];
    t21 = di2 & SNPj[p + 1];
    t22 = di2 & dj2;

    ft[0] += __popc(t00);
    ft[1] += __popc(t01);
    ft[2] += __popc(t02);
    ft[3] += __popc(t10);
    ft[4] += __popc(t11);
    ft[5] += __popc(t12);
    ft[6] += __popc(t20);
    ft[7] += __popc(t21);
    ft[8] += __popc(t22);

    // Phenotype 1
    SNPi = (unsigned int*) &dev_data_ones[i * 2];
    SNPj = (unsigned int*) &dev_data_ones[j * 2];

    #pragma unroll 1
    for(p = 0; p < 2 * PP_ones * num_snp - 2 * num_snp; p += 2 * num_snp)
    {
      di2 = ~(SNPi[p] | SNPi[p + 1]);
      dj2 = ~(SNPj[p] | SNPj[p + 1]);

      t00 = SNPi[p] & SNPj[p];
      t01 = SNPi[p] & SNPj[p + 1];
      t02 = SNPi[p] & dj2;
      t10 = SNPi[p + 1] & SNPj[p];
      t11 = SNPi[p + 1] & SNPj[p + 1];
      t12 = SNPi[p + 1] & dj2;
      t20 = di2 & SNPj[p];
      t21 = di2 & SNPj[p + 1];
      t22 = di2 & dj2;

      ft[9]  += __popc(t00);
      ft[10] += __popc(t01);
      ft[11] += __popc(t02);
      ft[12] += __popc(t10);
      ft[13] += __popc(t11);
      ft[14] += __popc(t12);
      ft[15] += __popc(t20);
      ft[16] += __popc(t21);
      ft[17] += __popc(t22);
    }
    p = 2 * PP_ones * num_snp - 2 * num_snp;
    di2 = ~(SNPi[p] | SNPi[p + 1]);
    dj2 = ~(SNPj[p] | SNPj[p + 1]);
    di2 = di2 & mask_ones;
    dj2 = dj2 & mask_ones;

    t00 = SNPi[p] & SNPj[p];
    t01 = SNPi[p] & SNPj[p + 1];
    t02 = SNPi[p] & dj2;
    t10 = SNPi[p + 1] & SNPj[p];
    t11 = SNPi[p + 1] & SNPj[p + 1];
    t12 = SNPi[p + 1] & dj2;
    t20 = di2 & SNPj[p];
    t21 = di2 & SNPj[p + 1];
    t22 = di2 & dj2;

    ft[9]  += __popc(t00);
    ft[10] += __popc(t01);
    ft[11] += __popc(t02);
    ft[12] += __popc(t10);
    ft[13] += __popc(t11);
    ft[14] += __popc(t12);
    ft[15] += __popc(t20);
    ft[16] += __popc(t21);
    ft[17] += __popc(t22);

    // compute score
    float score = 0.0f;

    #pragma unroll
    for(k = 0; k < 9; k++)
      score += gammafunction(ft[k] + ft[9 + k] + 1) -
               gammafunction(ft[k]) - gammafunction(ft[9 + k]);
    score = fabsf(score);
    if(score == 0.0f)
      score = FLT_MAX;
    dev_scores[tid] = score;
  }
}


int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <number of samples> <number of SNPs> <repeat>\n", argv[0]);
    return 1;
  }
  int i, j, x;
  int num_pac = atoi(argv[1]);  // #samples
  int num_snp = atoi(argv[2]);  // #SNPs
  int iteration = atoi(argv[3]);// #kernel run
  int block_snp = 64;

  srand(100);
  unsigned char *SNP_Data = mem_alloc<unsigned char>(64, num_pac * num_snp);
  unsigned char *Ph_Data = mem_alloc<unsigned char>(64, num_pac);

  // generate SNPs between 0 and 2
  for (i = 0; i < num_pac; i++)
    for(j = 0; j < num_snp; j++)
      SNP_Data[i * num_snp + j] = rand() % 3;

  // generate phenotype between 0 and 1
  for(int i = 0; i < num_pac; i++) Ph_Data[i] = rand() % 2;

  // transpose the SNP data
  unsigned char *SNP_Data_trans = mem_alloc<unsigned char>(64, num_pac * num_snp);

  for (i = 0; i < num_pac; i++) 
    for(j = 0; j < num_snp; j++) 
      SNP_Data_trans[j * num_pac + i] = SNP_Data[i * num_snp + j];

  int phen_ones = 0;
  for(i = 0; i < num_pac; i++)
    if(Ph_Data[i] == 1)
      phen_ones++;

  // transform SNP data to a binary format

  int PP_zeros = ceil((1.0*(num_pac - phen_ones))/32.0);
  int PP_ones = ceil((1.0*phen_ones)/32.0);

  unsigned int *bin_data_zeros = mem_alloc<unsigned int>(64, num_snp * PP_zeros * 2);
  unsigned int *bin_data_ones = mem_alloc<unsigned int>(64, num_snp * PP_ones * 2);
  memset(bin_data_zeros, 0, num_snp*PP_zeros*2*sizeof(unsigned int));
  memset(bin_data_ones, 0, num_snp*PP_ones*2*sizeof(unsigned int));

  for(i = 0; i < num_snp; i++)
  {
    int x_zeros = -1;
    int x_ones = -1;
    int n_zeros = 0;
    int n_ones = 0;

    for(j = 0; j < num_pac; j++){
      unsigned int temp = (unsigned int) SNP_Data_trans[i * num_pac + j];

      if(Ph_Data[j] == 1){
        if(n_ones%32 == 0){
          x_ones ++;
        }
        // apply 1 shift left to 2 components
        bin_data_ones[i * PP_ones * 2 + x_ones*2 + 0] <<= 1;
        bin_data_ones[i * PP_ones * 2 + x_ones*2 + 1] <<= 1;
        // insert '1' in correct component
        if(temp == 0 || temp == 1){
          bin_data_ones[i * PP_ones * 2 + x_ones*2 + temp ] |= 1;
        }
        n_ones ++;
      } else {
        if(n_zeros%32 == 0){
          x_zeros ++;
        }
        // apply 1 shift left to 2 components
        bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + 0] <<= 1;
        bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + 1] <<= 1;
        // insert '1' in correct component
        if(temp == 0 || temp == 1){
          bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + temp] |= 1;
        }
        n_zeros ++;
      }
    }
  }

  unsigned int mask_zeros = 0xFFFFFFFF;
  for(int x = num_pac - phen_ones; x < PP_zeros * 32; x++)
    mask_zeros = mask_zeros >> 1;

  unsigned int mask_ones = 0xFFFFFFFF;
  for(x = phen_ones; x < PP_ones * 32; x++)
    mask_ones = mask_ones >> 1;

  // transpose the binary data structures
  unsigned int* bin_data_ones_trans = mem_alloc<unsigned int>(64, num_snp * PP_ones * 2);

  for(i = 0; i < num_snp; i++)
    for(j = 0; j < PP_ones; j++)
    {
      bin_data_ones_trans[(j * num_snp + i) * 2 + 0] = bin_data_ones[(i * PP_ones + j) * 2 + 0];
      bin_data_ones_trans[(j * num_snp + i) * 2 + 1] = bin_data_ones[(i * PP_ones + j) * 2 + 1];
    }

  unsigned int* bin_data_zeros_trans = mem_alloc<unsigned int>(64, num_snp * PP_zeros * 2);

  for(i = 0; i < num_snp; i++)
    for(j = 0; j < PP_zeros; j++)
    {
      bin_data_zeros_trans[(j * num_snp + i) * 2 + 0] = bin_data_zeros[(i * PP_zeros + j) * 2 + 0];
      bin_data_zeros_trans[(j * num_snp + i) * 2 + 1] = bin_data_zeros[(i * PP_zeros + j) * 2 + 1];
    }

  float *scores = mem_alloc<float>(64, num_snp * num_snp);
  float *scores_ref = mem_alloc<float>(64, num_snp * num_snp);
  for(x = 0; x < num_snp * num_snp; x++) {
    scores[x] = scores_ref[x] = FLT_MAX;
  }

  unsigned int* d_data_zeros;
  cudaMalloc((void**)&d_data_zeros, num_snp*PP_zeros*2*sizeof(unsigned int));
  cudaMemcpy(d_data_zeros, bin_data_zeros_trans, 
             num_snp*PP_zeros*2*sizeof(unsigned int), cudaMemcpyHostToDevice);

  unsigned int* d_data_ones;
  cudaMalloc((void**)&d_data_ones, num_snp*PP_ones*2*sizeof(unsigned int));
  cudaMemcpy(d_data_ones, bin_data_ones_trans, 
             num_snp*PP_ones*2*sizeof(unsigned int), cudaMemcpyHostToDevice);

  float* d_scores;
  cudaMalloc((void**)&d_scores, num_snp*num_snp*sizeof(float));
  cudaMemcpy(d_scores, scores, sizeof(float) * num_snp * num_snp, cudaMemcpyHostToDevice);

  // setup kernel ND-range
  int num_snp_m = num_snp;
  while(num_snp_m % block_snp != 0) num_snp_m++;

  dim3 grid(num_snp_m / block_snp, num_snp_m);
  dim3 block(block_snp, 1);

  // epistasis detection kernel

  auto kstart = myclock::now();

  for (int i = 0; i < iteration; i++) {
    epi<<<grid, block>>>(d_data_zeros, d_data_ones, d_scores, num_snp, 
                         PP_zeros, PP_ones, mask_zeros, mask_ones);
  }

  cudaDeviceSynchronize();
  myduration ktime = myclock::now() - kstart;
  auto total_ktime = ktime.count();

  std::cout << "Average kernel execution time: "
            << total_ktime / iteration << " (s)" << std::endl;

  cudaMemcpy(scores, d_scores, sizeof(float) * num_snp * num_snp, cudaMemcpyDeviceToHost);

  int p1 = min_score(scores, num_snp, num_snp);

  reference (bin_data_zeros_trans, bin_data_ones_trans, scores_ref, num_snp, 
             PP_zeros, PP_ones, mask_zeros, mask_ones);

  int p2 = min_score(scores_ref, num_snp, num_snp);
  
  bool ok = (p1 == p2) && (fabsf(scores[p1] - scores_ref[p2]) < 1e-3f);
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  cudaFree(d_data_zeros);
  cudaFree(d_data_ones);
  cudaFree(d_scores);

  mem_free(bin_data_zeros);
  mem_free(bin_data_ones);
  mem_free(bin_data_zeros_trans);
  mem_free(bin_data_ones_trans);
  mem_free(scores);
  mem_free(scores_ref);
  mem_free(SNP_Data);
  mem_free(SNP_Data_trans);
  mem_free(Ph_Data);
  if (!ok) return 1;
  return 0;
}
