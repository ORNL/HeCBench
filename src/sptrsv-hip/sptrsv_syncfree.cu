#ifndef _SPTRSV_SYNCFREE_
#define _SPTRSV_SYNCFREE_

#include <chrono>
#include <hip/hip_runtime.h>
#include "sptrsv.h"
#include <cstdlib>

// reference
// https://stackoverflow.com/questions/32341081/how-to-have-atomic-load-in-cuda

// addr must be aligned properly.
__device__ int atomic_load(const int *addr)
{
  const volatile int *vaddr = addr; // volatile to bypass cache
  //__threadfence(); // for seq_cst loads. Remove for acquire semantics.
  const int value = *vaddr;
  // fence to ensure that dependent reads are correctly ordered
  __threadfence(); 
  return value; 
}

// addr must be aligned properly.
__device__ void atomic_store(int *addr, int value)
{
  volatile int *vaddr = addr; // volatile to bypass cache
  // fence to ensure that previous non-atomic stores are visible to other threads
  __threadfence(); 
  *vaddr = value;
}

__global__
void sptrsv_mix(
    const int        *__restrict__ csrRowPtr,
    const int        *__restrict__ csrColIdx,
    const VALUE_TYPE *__restrict__ csrVal,
    int              *__restrict__ get_value,
    const int        m,
    const VALUE_TYPE *__restrict__ b,
    VALUE_TYPE       *__restrict__ x,
    const int        *__restrict__ warp_num,
    const int        Len)
{
  const int local_id = threadIdx.x;
  const int global_id = blockIdx.x * blockDim.x + local_id;
  const int warp_id = global_id/WARP_SIZE;

  int row;
  __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK*WARP_SIZE];

  if(warp_id>=(Len-1)) return;

  const int lane_id = (WARP_SIZE - 1) & local_id;

  if(warp_num[warp_id+1]>(warp_num[warp_id]+1))
  {
    //thread
    row =warp_num[warp_id]+lane_id;
    if(row>=m) return;

    int col,j,i;
    VALUE_TYPE xi;
    VALUE_TYPE left_sum=0;
    i=row;
    j=csrRowPtr[i];

    while(j<csrRowPtr[i+1])
    {
      col=csrColIdx[j];
      if(atomic_load(&get_value[col])==1)
      {
        left_sum+=csrVal[j]*x[col];
        j++;
        col=csrColIdx[j];
      }
      if(i==col)
      {
        xi = (b[i] - left_sum) / csrVal[csrRowPtr[i+1]-1];
        x[i] = xi;
        atomic_store(&get_value[i], 1);
        j++;
      }
    }
  }
  else
  {
    row = warp_num[warp_id];
    if(row>=m)
      return;

    int col,j=csrRowPtr[row]  + lane_id;
    VALUE_TYPE xi,sum=0;
    while(j < (csrRowPtr[row+1]-1))
    {
      col=csrColIdx[j];
      if(atomic_load(&get_value[col])==1)
      {
        sum += x[col] * csrVal[j];
        j += WARP_SIZE;
      }
    }

    s_left_sum[local_id]=sum;

    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
    {
      if(lane_id < offset)
      {
        s_left_sum[local_id] += s_left_sum[local_id+offset];
      }
    }

    if (!lane_id)
    {
      xi = (b[row] - s_left_sum[local_id]) / csrVal[csrRowPtr[row+1]-1];
      x[row]=xi;
      atomic_store(&get_value[row], 1);
    }
  }
}

int sptrsv_syncfree (
    const int           repeat,
    const int           *csrRowPtr,
    const int           *csrColIdx,
    const VALUE_TYPE    *csrVal,
    const int           m,
    const int           n,
    const int           nnz,
    VALUE_TYPE          *x,
    const VALUE_TYPE    *b,
    const VALUE_TYPE    *x_ref)
{
  if (m != n)
  {
    printf("This is not a square matrix, return.\n");
    return -1;
  }

  int *warp_num=(int *)malloc((m+1)*sizeof(int));
  memset (warp_num, 0, sizeof(int)*(m+1));

  memset(x, 0, m * sizeof(VALUE_TYPE));

  double warp_occupy=0,element_occupy=0;
  int Len;

  for(int i=0;i<repeat;i++)
  {
    matrix_warp4(m,n,nnz,csrRowPtr,csrColIdx,csrVal,10,&Len,warp_num,&warp_occupy,&element_occupy);
  }

  // Matrix L
  int* d_csrRowPtr;
  hipMalloc((void**)&d_csrRowPtr, sizeof(int)*(n+1));
  hipMemcpy(d_csrRowPtr, csrRowPtr, sizeof(int)*(n+1), hipMemcpyHostToDevice);

  int* d_csrColIdx;
  hipMalloc((void**)&d_csrColIdx, sizeof(int)*nnz);
  hipMemcpy(d_csrColIdx, csrColIdx, sizeof(int)*nnz, hipMemcpyHostToDevice);

  VALUE_TYPE* d_csrVal;
  hipMalloc((void**)&d_csrVal, sizeof(VALUE_TYPE)*nnz);
  hipMemcpy(d_csrVal, csrVal, sizeof(VALUE_TYPE)*nnz, hipMemcpyHostToDevice);

  // Vector b
  VALUE_TYPE* d_b;
  hipMalloc((void**)&d_b, sizeof(VALUE_TYPE)*m);
  hipMemcpy(d_b, b, sizeof(VALUE_TYPE)*m, hipMemcpyHostToDevice);

  // Vector x
  VALUE_TYPE* d_x;
  hipMalloc((void**)&d_x, sizeof(VALUE_TYPE)*n);

  int *d_get_value;
  hipMalloc((void**)&d_get_value, sizeof(int)*m);

  int* d_warp_num;
  hipMalloc((void**)&d_warp_num, sizeof(int)*Len);
  hipMemcpy(d_warp_num, warp_num, sizeof(int)*Len, hipMemcpyHostToDevice);

  int num_threads = WARP_PER_BLOCK * WARP_SIZE;
  int num_blocks = ceil ((double)((Len-1)*WARP_SIZE) / (double)(num_threads));

  double time = 0.0;

  for (int i = 0; i <= repeat; i++)
  {
    hipMemset(d_get_value, 0, sizeof(int)*m);
    hipMemcpy(d_x, x, sizeof(VALUE_TYPE)*n, hipMemcpyHostToDevice);

    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    hipLaunchKernelGGL(sptrsv_mix, num_blocks, num_threads, 0, 0, 
        d_csrRowPtr, d_csrColIdx, d_csrVal, d_get_value,
        m, d_b, d_x, d_warp_num, Len);

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    if (i > 0)
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  hipMemcpy(x, d_x, sizeof(VALUE_TYPE)*n, hipMemcpyDeviceToHost);

  // validate x
  double accuracy = 1e-4;
  double ref = 0.0;
  double res = 0.0;
  int error=0;
  const int rhs = 1;

  for (int i = 0; i < n * rhs; i++)
  {
    ref += abs(x_ref[i]);
    res += abs(x[i] - x_ref[i]);
    if(x[i] != x_ref[i]  && error<10)
    {
      // printf("%d %f %f\n",i,x[i],x_ref[i]);
      error++;
    }
  }
  res = ref == 0 ? res : res / ref;

  printf("|x-xref|/|xref| = %8.2e\n", res);

  printf("%s\n", (res < accuracy) ? "PASS" : "FAIL");
  if (!(res < accuracy)) exit(1);

  free(warp_num);
  hipFree(d_csrRowPtr);
  hipFree(d_csrColIdx);
  hipFree(d_csrVal);
  hipFree(d_get_value);
  hipFree(d_b);
  hipFree(d_x);
  hipFree(d_warp_num);
  return 0;
}

#endif
