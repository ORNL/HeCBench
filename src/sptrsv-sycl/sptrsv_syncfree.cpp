#ifndef _SPTRSV_SYNCFREE_
#define _SPTRSV_SYNCFREE_

#include <chrono>
#include <sycl/sycl.hpp>
#include "sptrsv.h"
#include <cstdlib>

int atomicLoad(const int *addr)
{
  const volatile int *vaddr = addr; // volatile to bypass cache
  const int value = *vaddr;
  // fence to ensure that dependent reads are correctly ordered
  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
  return value;
}

// addr must be aligned properly.
void atomicStore(int *addr, int value)
{
  volatile int *vaddr = addr; // volatile to bypass cache
  // fence to ensure that previous non-atomic stores are visible to other threads
  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
  *vaddr = value;
}

int sptrsv_syncfree (
    const int           repeat,
    const int           *csrRowPtr,
    const int           *csrColIdx,
    const VALUE_TYPE    *csrVal,
    const int            m,
    const int            n,
    const int            nnz,
    VALUE_TYPE    *x,
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Matrix L
  int *d_csrRowPtr= sycl::malloc_device<int>(n+1, q);
  q.memcpy(d_csrRowPtr, csrRowPtr, sizeof(int)*(n+1));

  int *d_csrColIdx= sycl::malloc_device<int>(nnz, q);
  q.memcpy(d_csrColIdx, csrColIdx, sizeof(int)*nnz);

  VALUE_TYPE *d_csrVal= sycl::malloc_device<VALUE_TYPE>(nnz, q);
  q.memcpy(d_csrVal, csrVal, sizeof(VALUE_TYPE)*nnz);

  // Vector b
  VALUE_TYPE *d_b = sycl::malloc_device<VALUE_TYPE>(m, q);
  q.memcpy(d_b, b, sizeof(VALUE_TYPE)*m);

  // Vector x
  VALUE_TYPE *d_x = sycl::malloc_device<VALUE_TYPE>(n, q);

  int *d_get_value = sycl::malloc_device<int>(m, q);

  int *d_warp_num = sycl::malloc_device<int>(Len, q);
  q.memcpy(d_warp_num, warp_num, sizeof(int)*Len);

  int num_threads = WARP_PER_BLOCK * WARP_SIZE;
  int num_blocks = ceil ((double)((Len-1)*WARP_SIZE) / (double)(num_threads));

  sycl::range<1> gws (num_blocks * num_threads);
  sycl::range<1> lws (num_threads);

  double time = 0.0;

  for (int i = 0; i <= repeat; i++)
  {
    q.memset(d_get_value, 0, sizeof(int)*m);
    q.memcpy(d_x, x, sizeof(VALUE_TYPE)*n);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<VALUE_TYPE, 1> s_left_sum(lws, cgh);
      cgh.parallel_for<class sptrsv_mix>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {

        const int global_id = item.get_global_id(0);
        const int warp_id = global_id/WARP_SIZE;
        const int local_id = item.get_local_id(0);

        int row;

        if(warp_id>=(Len-1)) return;

        const int lane_id = (WARP_SIZE - 1) & local_id;

        if(d_warp_num[warp_id+1]>(d_warp_num[warp_id]+1))
        {
          //thread
          row =d_warp_num[warp_id]+lane_id;
          if(row>=m) return;

          int col,j,i;
          VALUE_TYPE xi;
          VALUE_TYPE left_sum=0;
          i=row;
          j=d_csrRowPtr[i];

          while(j<d_csrRowPtr[i+1])
          {
            col=d_csrColIdx[j];
            if(atomicLoad(&d_get_value[col])==1)
            {
              left_sum+=d_csrVal[j]*d_x[col];
              j++;
              col=d_csrColIdx[j];
            }
            if(i==col)
            {
              xi = (d_b[i] - left_sum) / d_csrVal[d_csrRowPtr[i+1]-1];
              d_x[i] = xi;
              atomicStore(&d_get_value[i], 1);
              j++;
            }
          }
        }
        else
        {
          row = d_warp_num[warp_id];
          if(row>=m)
            return;

          int col,j=d_csrRowPtr[row]  + lane_id;
          VALUE_TYPE xi,sum=0;
          while(j < (d_csrRowPtr[row+1]-1))
          {
            col=d_csrColIdx[j];
            if(atomicLoad(&d_get_value[col])==1)
            {
              sum += d_x[col] * d_csrVal[j];
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
            xi = (d_b[row] - s_left_sum[local_id]) / d_csrVal[d_csrRowPtr[row+1]-1];
            d_x[row]=xi;
            atomicStore(&d_get_value[i], 1);
          }
        }
      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    if (i > 0)
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(x, d_x, sizeof(VALUE_TYPE)*n).wait();

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

  free(warp_num);
  sycl::free(d_csrRowPtr, q);
  sycl::free(d_csrColIdx, q);
  sycl::free(d_csrVal, q);
  sycl::free(d_get_value, q);
  sycl::free(d_b, q);
  sycl::free(d_x, q);
  sycl::free(d_warp_num, q);

  if (!(res < accuracy)) return 1;
  return 0;
}

#endif
