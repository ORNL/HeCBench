#ifndef _SPTRSV_SYNCFREE_
#define _SPTRSV_SYNCFREE_

#include <chrono>
#include <omp.h>
#include "sptrsv.h"
#include <cstdlib>

#pragma omp declare target
int atomicLoad(const int *addr)
{
  const volatile int *vaddr = addr; // volatile to bypass cache
  int value;

  #pragma omp atomic read
  value = *vaddr;

  // fence to ensure that dependent reads are correctly ordered
  //#pragma omp flush

  return value; 
}

// addr must be aligned properly.
void atomicStore(int *addr, int value)
{
  volatile int *vaddr = addr; // volatile to bypass cache

  // fence to ensure that previous non-atomic stores are visible to other threads
  //#pragma omp flush

  #pragma omp atomic write
  *vaddr = value;
}

#pragma omp end declare target

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

  int *get_value = (int *)malloc(m * sizeof(int));

  double warp_occupy=0,element_occupy=0;
  int Len;

  for(int i=0;i<repeat;i++)
  {
    matrix_warp4(m,n,nnz,csrRowPtr,csrColIdx,csrVal,10,&Len,warp_num,&warp_occupy,&element_occupy);
  }


  #pragma omp target data map(to: csrRowPtr[0:m+1],\
                                  csrColIdx[0:nnz],\
                                  csrVal[0:nnz],\
                                  b[0:m],\
                                  warp_num[0:Len])\
                          map(alloc: x[0:n], get_value[0:m]) 
  {
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil ((double)((Len-1)*WARP_SIZE) / (double)(num_threads));

    double time = 0.0;

    for (int i = 0; i <= repeat; i++)
    {
      memset(x, 0, n * sizeof(VALUE_TYPE));
      memset(get_value, 0, m * sizeof(int));
      #pragma omp target update to (x[0:n])
      #pragma omp target update to (get_value[0:m])

      auto start = std::chrono::steady_clock::now();

      #pragma omp target teams num_teams(num_blocks) thread_limit(num_threads)
      {
        VALUE_TYPE s_left_sum[WARP_PER_BLOCK*WARP_SIZE];
        #pragma omp parallel 
        {
          const int local_id = omp_get_thread_num();
          const int global_id = omp_get_team_num() * num_threads + local_id;
          const int warp_id = global_id/WARP_SIZE;

          int row;

          if(warp_id < (Len-1)) {

            const int lane_id = (WARP_SIZE - 1) & local_id;

            if(warp_num[warp_id+1]>(warp_num[warp_id]+1))
            {
              //thread
              row =warp_num[warp_id]+lane_id;
              if(row < m) {

                int col,j,i;
                VALUE_TYPE xi;
                VALUE_TYPE left_sum=0;
                i=row;
                j=csrRowPtr[i];

                while(j<csrRowPtr[i+1])
                {
                  col=csrColIdx[j];
                  if(atomicLoad(&get_value[col])==1)
                  {
                    left_sum+=csrVal[j]*x[col];
                    j++;
                    col=csrColIdx[j];
                  }
                  if(i==col)
                  {
                    xi = (b[i] - left_sum) / csrVal[csrRowPtr[i+1]-1];
                    x[i] = xi;
                    get_value[i] = 1;
                    atomicStore(&get_value[i], 1);
                    j++;
                  }
                }
              }
            }
            else
            {
              row = warp_num[warp_id];
              if(row < m) {

                int col,j=csrRowPtr[row]  + lane_id;
                VALUE_TYPE xi,sum=0;
                while(j < (csrRowPtr[row+1]-1))
                {
                  col=csrColIdx[j];
                  if(atomicLoad(&get_value[col])==1)
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
                  atomicStore(&get_value[i], 1);
                }
              }
            }
          }
        }
      }

      auto end = std::chrono::steady_clock::now();
      if (i > 0)
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
    #pragma omp target update from (x[0:m])
  }

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

  free(get_value);
  free(warp_num);

  if (!(res < accuracy)) return 1;
  return 0;
}

#endif
