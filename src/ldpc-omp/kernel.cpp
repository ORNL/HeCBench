/*  Copyright (c) 2011-2016, Robert Wang, email: robertwgh (at) gmail.com
    All rights reserved. https://github.com/robertwgh/cuLDPC

    CUDA implementation of LDPC decoding algorithm.

    The details of implementation can be found from the following papers:
    1. Wang, G., Wu, M., Sun, Y., & Cavallaro, J. R. (2011, June). A massively parallel implementation of QC-LDPC decoder on GPU. In Application Specific Processors (SASP), 2011 IEEE 9th Symposium on (pp. 82-85). IEEE.
    2. Wang, G., Wu, M., Yin, B., & Cavallaro, J. R. (2013, December). High throughput low latency LDPC decoding on GPU for SDR systems. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE (pp. 1258-1261). IEEE.

    The current release is close to the GlobalSIP2013 paper.

Created:   10/1/2010
Revision:  08/01/2013
04/20/2016 prepare for release on Github.
11/26/2017 cleanup and comments by Albin Severinson, albin (at) severinson.org
 */

#include "LDPC.h"


// phi(x) = log((e^x + 1) / (e^x - 1)) = log1p(2 / (e^x - 1)).
// phi is its own inverse, so the log-SPA check-node magnitude is
// |R_i| = phi(sum over j != i of phi(|Q_j|)). Clamping the argument away from
// zero keeps phi(0) from returning infinity and caps |R| at about 14.5.
#pragma omp declare target
inline float phi(float x)
{
  return log1pf(2.0f / expm1f(fmaxf(x, 1e-6f)));
}
#pragma omp end declare target

// MinSum selects the check-node update rule: normalized min-sum when true,
// log-SPA when false. It is a template parameter rather than a macro so that
// both rules are compiled into the same binary and can be run in turn.
//
// The log-SPA branch needs every phi(|Q_j|) again in the second recursion and
// so holds them in a small private array. Its loops are bounded by a
// compile-time constant rather than by s, which is what keeps that array in
// registers; with a runtime bound the index forces it into local memory.

// Kernel 1
template <bool MinSum>
void ldpc_cnp_kernel_1st_iter(
    const float * dev_llr,
    float * dev_dt,
    float * dev_R,
    const char * dev_h_element_count1,
    const h_element * dev_h_compact1)
{
  #pragma omp target teams num_teams(BLK_ROW*MCW) thread_limit(THREADS_PER_BLOCK)
  {
    #pragma omp parallel
    {
      int iSubRow = omp_get_thread_num() % BLOCK_SIZE_X;

#if MODE == WIFI
      if(iSubRow < Z) {
#endif
      int iCW     = omp_get_thread_num() / BLOCK_SIZE_X;
      int iMCW    = omp_get_team_num() / BLK_ROW;
      int iBlkRow = omp_get_team_num() % BLK_ROW;

      //iSubRow = threadIdx.x;
      //int iCW = threadIdx.y; // index of CW in a MCW
      //int iBlkRow = blockIdx.x;
      //int iMCW = blockIdx.y; // index of MCW

      int iCurrentCW = iMCW * CW + iCW;

      //for step 1: update dt
      int iBlkCol; // block col in h_base
      int iCol; // overall col index in h_base
      int offsetR;

      int size_llr_CW = COL; // size of one llr CW block
      int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block
      int shift_t;

      char Q_sign = 0;
      char sq;
      float Q, Q_abs;
      float R_temp;

      float sign = 1.0f;

      // For 2-min algorithm.
      float rmin1 = 1000.0f;
      float rmin2 = 1000.0f;
      char idx_min = 0;

      // For log-SPA.
      float phi_sum = 0.0f;
      float phi_q[NON_EMPTY_ELMENT];

      h_element h_element_t;
      int s = dev_h_element_count1[iBlkRow];
      offsetR = size_R_CW * iCurrentCW + iBlkRow * Z + iSubRow;

      // The 1st recursion
      for(int i = 0; i < (MinSum ? s : NON_EMPTY_ELMENT); i++) // loop through all the ZxZ sub-blocks in a row
      {
        if (i >= s) break;
        h_element_t = dev_h_compact1[i * H_COMPACT1_ROW + iBlkRow];

        iBlkCol = h_element_t.y;
        shift_t = h_element_t.value;

        shift_t = (iSubRow + shift_t);
        if(shift_t >= Z) shift_t = shift_t - Z;

        iCol = iBlkCol * Z + shift_t;

        Q = dev_llr[size_llr_CW * iCurrentCW + iCol];// - R_temp;
        Q_abs = fabsf(Q);
        sq = Q < 0;

        // quick version
        sign = sign * (1 - sq * 2);
        Q_sign |= sq << i;

        if constexpr (MinSum)
        {
          if (Q_abs < rmin1)
          {
            rmin2 = rmin1;
            rmin1 = Q_abs;
            idx_min = i;
          } else if (Q_abs < rmin2)
          {
            rmin2 = Q_abs;
          }
        }
        else
        {
          phi_q[i] = phi(Q_abs);
          phi_sum += phi_q[i];
        }
      }

      // The 2nd recursion
      for(int i = 0; i < (MinSum ? s : NON_EMPTY_ELMENT); i ++)
      {
        if (i >= s) break;
        sq = 1 - 2 * ((Q_sign >> i) & 0x01);
        if constexpr (MinSum)
          // v0: Best performance so far. 0.75f is the value of alpha.
          R_temp = 0.75f * sign * sq * (i != idx_min ? rmin1 : rmin2);
        else
          R_temp = sign * sq * phi(phi_sum - phi_q[i]);

        // write results to global memory
        h_element_t = dev_h_compact1[i * H_COMPACT1_ROW + iBlkRow];
        int addr_temp = offsetR + h_element_t.y * ROW;
        dev_dt[addr_temp] = R_temp;// - R1[i]; // compute the dt value for current llr.
        dev_R[addr_temp] = R_temp; // update R, R=R'.
      }
#if MODE == WIFI
      }
#endif
    }
  }
}

// Kernel_1
template <bool MinSum>
void ldpc_cnp_kernel(
    const float * dev_llr,
    float * dev_dt,
    float * dev_R,
    const char * dev_h_element_count1,
    const h_element * dev_h_compact1)
{
  #pragma omp target teams num_teams(BLK_ROW*MCW) thread_limit(THREADS_PER_BLOCK)
  {
    float RCache[BLOCK_SIZE_X * CW * NON_EMPTY_ELMENT];
    #pragma omp parallel
    {
      int iSubRow = omp_get_thread_num() % BLOCK_SIZE_X;

#if MODE == WIFI
      if(iSubRow < Z) {
#endif

  // Define cache for R: Rcache[NON_EMPTY_ELMENT][nThreadPerBlock]
  // extern means that the memory is allocated dynamically at run-time
      int iCW     = omp_get_thread_num() / BLOCK_SIZE_X;
      int iMCW    = omp_get_team_num() / BLK_ROW;
      int iBlkRow = omp_get_team_num() % BLK_ROW;

      //int iRCacheLine = threadIdx.y * blockDim.x + threadIdx.x;
      int iRCacheLine = iCW * BLOCK_SIZE_X + iSubRow;


      int iCol; // overall col index in h_base
      int iBlkCol;
      int offsetR;

      int iCurrentCW = iMCW * CW + iCW;

      int size_llr_CW = COL; // size of one llr CW block
      int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block

      //float R1[NON_EMPTY_ELMENT];
      int shift_t;

      char Q_sign = 0;
      char sq;
      float Q, Q_abs;
      float R_temp;

      float sign = 1.0f;

      // For 2-min algorithm.
      float rmin1 = 1000.0f;
      float rmin2 = 1000.0f;
      char idx_min = 0;

      // For log-SPA.
      float phi_sum = 0.0f;
      float phi_q[NON_EMPTY_ELMENT];

      h_element h_element_t;
      int s = dev_h_element_count1[iBlkRow];
      offsetR = size_R_CW * iCurrentCW + iBlkRow * Z + iSubRow;

      // The 1st recursion
      // TODO: Is s always the same? If so we can unroll the loop with #pragma unroll
      for(int i = 0; i < (MinSum ? s : NON_EMPTY_ELMENT); i++) // loop through all the ZxZ sub-blocks in a row
      {
        if (i >= s) break;
        h_element_t = dev_h_compact1[i * H_COMPACT1_ROW + iBlkRow];

        iBlkCol = h_element_t.y;
        shift_t = h_element_t.value;

        shift_t = (iSubRow + shift_t);
        if(shift_t >= Z) shift_t = shift_t - Z;

        iCol = iBlkCol * Z + shift_t;

        R_temp = dev_R[offsetR + iBlkCol * ROW];

        RCache[i * THREADS_PER_BLOCK + iRCacheLine] =  R_temp;

        Q = dev_llr[size_llr_CW * iCurrentCW + iCol] - R_temp;
        Q_abs = fabsf(Q);

        sq = Q < 0;
        sign = sign * (1 - sq * 2);
        Q_sign |= sq << i;

        if constexpr (MinSum)
        {
          if (Q_abs < rmin1)
          {
            rmin2 = rmin1;
            rmin1 = Q_abs;
            idx_min = i;
          } else if (Q_abs < rmin2)
          {
            rmin2 = Q_abs;
          }
        }
        else
        {
          phi_q[i] = phi(Q_abs);
          phi_sum += phi_q[i];
        }
      }

      #pragma omp barrier

      // The 2nd recursion
      //#pragma unroll
      for(int i = 0; i < (MinSum ? s : NON_EMPTY_ELMENT); i ++)
      {
        if (i >= s) break;
        sq = 1 - 2 * ((Q_sign >> i) & 0x01);
        if constexpr (MinSum)
          R_temp = 0.75f * sign * sq * (i != idx_min ? rmin1 : rmin2);
        else
          R_temp = sign * sq * phi(phi_sum - phi_q[i]);

        // write results to global memory
        h_element_t = dev_h_compact1[i * H_COMPACT1_ROW + iBlkRow];
        int addr_temp = h_element_t.y * ROW + offsetR;
        dev_dt[addr_temp] = R_temp - RCache[i * THREADS_PER_BLOCK + iRCacheLine];
        dev_R[addr_temp] = R_temp; // update R, R=R'.
      }
#if MODE == WIFI
      }
#endif
    }
  }
}

// Kernel 2: VNP processing
void
ldpc_vnp_kernel_normal(
    float * dev_llr,
    float * dev_dt,
    const char *dev_h_element_count2,
    const h_element *dev_h_compact2)
{

  #pragma omp target teams num_teams(BLK_COL*MCW) thread_limit(THREADS_PER_BLOCK)
  {
    #pragma omp parallel
    {

      int iSubCol = omp_get_thread_num() % BLOCK_SIZE_X;

#if MODE == WIFI
      if(iSubCol < Z) {
#endif

      int iCW     = omp_get_thread_num() / BLOCK_SIZE_X;
      int iMCW    = omp_get_team_num() / BLK_COL;
      int iBlkCol = omp_get_team_num() % BLK_COL;
      int iBlkRow;
      int iCurrentCW = iMCW * CW + iCW;

      int iRow;
      int iCol;

      int shift_t, sf;
      int llr_index;
      float APP;

      h_element h_element_t;

      int size_llr_CW = COL; // size of one llr CW block
      int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block

      // update all the llr values
      iCol = iBlkCol * Z + iSubCol;
      llr_index = size_llr_CW * iCurrentCW + iCol;

      APP = dev_llr[llr_index];
      int offsetDt = size_R_CW * iCurrentCW + iBlkCol * ROW;

      for(int i = 0; i < dev_h_element_count2[iBlkCol]; i++)
      {
        h_element_t = dev_h_compact2[i * H_COMPACT2_COL + iBlkCol];

        shift_t = h_element_t.value;
        iBlkRow = h_element_t.x;

        sf = iSubCol - shift_t;
        if(sf < 0) sf = sf + Z;

        iRow = iBlkRow * Z + sf;
        APP = APP + dev_dt[offsetDt + iRow];
      }
      // write back to device global memory
      dev_llr[llr_index] = APP;

      // No hard decision for non-last iteration.
#if MODE == WIFI
      }
#endif
    }
  }
}

// Kernel: VNP processing for the last iteration.
void ldpc_vnp_kernel_last_iter(
    const float * dev_llr,
    const float * dev_dt,
    int * dev_hd,
    const char *dev_h_element_count2,
    const h_element *dev_h_compact2)
{
  #pragma omp target teams num_teams(BLK_COL*MCW) thread_limit(THREADS_PER_BLOCK)
  {
    #pragma omp parallel
    {

      int iSubCol = omp_get_thread_num() % BLOCK_SIZE_X;

#if MODE == WIFI
      if(iSubCol < Z) {
#endif

      int iCW     = omp_get_thread_num() / BLOCK_SIZE_X;
      int iMCW    = omp_get_team_num() / BLK_COL;
      int iBlkCol = omp_get_team_num() % BLK_COL;
      int iBlkRow;

      int iCurrentCW = iMCW * CW + iCW;
      int iRow;
      int iCol;

      int shift_t, sf;
      int llr_index;
      float APP;

      h_element h_element_t;


      int size_llr_CW = COL; // size of one llr CW block
      int size_R_CW = ROW * BLK_COL;  // size of one R/dt CW block

      // update all the llr values
      iCol = iBlkCol * Z + iSubCol;
      llr_index = size_llr_CW * iCurrentCW + iCol;

      APP = dev_llr[llr_index];

      int offsetDt = size_R_CW * iCurrentCW + iBlkCol * ROW;

      for(int i = 0; i < dev_h_element_count2[iBlkCol]; i ++)
      {
        h_element_t = dev_h_compact2[i * H_COMPACT2_COL + iBlkCol];

        shift_t = h_element_t.value;
        iBlkRow = h_element_t.x;

        sf = iSubCol - shift_t;
        if(sf < 0) sf = sf + Z;

        iRow = iBlkRow * Z + sf;
        APP = APP + dev_dt[offsetDt + iRow];
      }

      // For the last iteration, we don't need to write intermediate results to
      // global memory. Instead, we directly make a hard decision.
      if(APP > 0)
        dev_hd[llr_index] = 0;
      else
        dev_hd[llr_index] = 1;
#if MODE == WIFI
      }
#endif
    }
  }
}
