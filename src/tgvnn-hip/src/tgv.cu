/*
  This file is part of the TGV package (https://github.com/chixindebaoyu/tgvnn).

  The MIT License (MIT)

  Copyright (c) Dong Wang

  Permission is hereby granted, free of charge, to any person obtaining a # copy
  of this software and associated documentation files (the "Software"), to # deal
  in the Software without restriction, including without limitation the # rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or # sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included # in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS # OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL # THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING # FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS # IN THE
  SOFTWARE.
*/

#include "tgv.h"

//Configuration parameters
int blocksize = 128;
int gridsize = 128;

//HIP global variables
hipStream_t stream[2];

hipblasStatus_t hipblas_status;
hipblasHandle_t hipblas_handle;

hipsolverStatus_t hipsolver_status;
hipsolverHandle_t hipsolver_handle;

hipfftHandle fft2_plan;

// Useful kernels
void
gpuAssert(hipError_t code, const char *file, int line)
{
    if (code != hipSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code),
            file, line);
        getchar();
        exit(code);
    }
}
#define cuTry(ans) gpuAssert((ans), __FILE__, __LINE__);

void *
mallocAssert(const size_t n, const char *file, int line)
{
    void *p = malloc(n);
    if (p == NULL)
    {
        printf("Malloc failed from %s:%d", file, line);
        abort();
    }
    return p;
}
#define safe_malloc(n) mallocAssert(n, __FILE__, __LINE__);

void
freeAssert(void *p, const char *file, int line)
{
    if (p == NULL)
    {
        printf("Free failed from %s:%d", file, line);
        abort();
    }
}
#define safe_free(x) freeAssert(x, __FILE__, __LINE__);

void
print_usage()
{
    fprintf(stderr, "Dynamic MRI reconstruction using Primal-Dual algorithm\n");
    fprintf(stderr, "Decomposition of second order TGV and nuclear norm\n");
    fprintf(stderr, "Usage: runtgv [OPTION] <img.ra> <mask.ra>\n");
    fprintf(stderr, "\t-o, --output <output.ra>\t recon output RA file\n");
    fprintf(stderr, "\t-i, --iter n\t\t\t iteration number\n");
    fprintf(stderr, "\t-a, --alpha n\t\t\t parameter for TGV\n");
    fprintf(stderr, "\t-b, --beta n\t\t\t parameter for nuclear norm\n");
    fprintf(stderr, "\t-s, --sigma n\t\t\t dual stepsize\n");
    fprintf(stderr, "\t-t, --tau n\t\t\t primal stepsize\n");
    fprintf(stderr, "\t-m, --mu n\t\t\t temporal stepsize\n");
    fprintf(stderr, "\t-G, --gridsize n\t\t set GPU gridsize\n");
    fprintf(stderr, "\t-B, --blocksize n\t\t set GPU blocksize\n");
    fprintf(stderr, "\t-h\t\t\t\t show this help\n");
}

void
save_rafile(float2 *h_out, const char *out_path,
  const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim3)
{
    // save out
    ra_t out;
    out.flags = 0;
    out.eltype = RA_TYPE_COMPLEX;
    out.elbyte = sizeof(float2);
    out.size = dim0 * dim1 * dim2 * dim3 * sizeof(float2);
    out.ndims = 4;
    out.dims = (uint64_t *) safe_malloc(out.ndims * sizeof(uint64_t));
    out.dims[0] = dim0;
    out.dims[1] = dim1;
    out.dims[2] = dim2;
    out.dims[3] = dim3;
    out.data = (uint8_t *) h_out;
    ra_write(&out, out_path);
    ra_free(&out);
}

__global__ void
scaledata (float2 *d_array, const size_t array_size, const float factor)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < array_size;
      id += blockDim.x * gridDim.x)
    {
      d_array[id] *= factor;
    }
}

float
compute_alpha (float alpha0, float alpha1, int iter, int index)
{
  return expf(float(index)/float(iter)*logf(alpha1) +
          (float(iter)-float(index))/float(iter)*logf(alpha0));
}

float
compute_maxmag (float2 *d_array, const size_t array_size)
{
  int max_idx;
  hipblas_status = hipblasIcamax(hipblas_handle, array_size, d_array, 1, &max_idx);
  if (hipblas_status != HIPBLAS_STATUS_SUCCESS) {
     printf ("CUBLAS Scnrm2 failed:%d\n", hipblas_status);
     exit(EXIT_FAILURE);
  }

  float2 *h_tmp;
  h_tmp = (float2*)safe_malloc(sizeof(float2));
  cuTry(hipMemcpyAsync(h_tmp, d_array+max_idx-1, sizeof(float2),
          hipMemcpyDeviceToHost));

  return abs(h_tmp[0].x);
}

__global__ void
arrayadd (float2 *array_c, float2 *array_a, float2 *array_b,
  const size_t array_size, float alpha, float beta)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < array_size;
      id += blockDim.x * gridDim.x)
    {
      array_c[id] = alpha*array_a[id] + beta*array_b[id];
    }
}

__global__ void
arrayadd (float *array_c, float *array_a, float *array_b,
  const size_t array_size, float alpha, float beta)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < array_size;
      id += blockDim.x * gridDim.x)
    {
      array_c[id] = alpha*array_a[id] + beta*array_b[id];
    }
}

__global__ void
arraydot (float2 *array_c, float2 *array_a, float2 *array_b,
  const size_t array_size, float alpha = 1.f)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < array_size;
      id += blockDim.x * gridDim.x)
    {
      array_c[id] = alpha*array_a[id]*array_b[id];
    }
}

__global__ void
arrayreal (float2 *d_out, float2 *d_in, size_t array_size)
{
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < array_size;
    id += blockDim.x * gridDim.x)
    {
      d_out[id] = make_float2(fmaxf(d_in[id].x, 0.f));
    }
}

void
fft2_init(const int rows, const int cols, const int ndyn)
{
  // setup FFT plan, ndyn = # timepoints
  const int rank = 2;
  int idist = 1, odist = 1, istride = ndyn, ostride = ndyn;
  int n[rank] = {cols, rows};

  hipfftPlanMany(&fft2_plan, rank, n,
      NULL, istride, idist,
      NULL, ostride, odist,
      HIPFFT_C2C, ndyn);
}

void
forward (float2 *d_out, float2 *d_in, float2 *d_mask,
  const size_t N, const size_t rows, const size_t cols)
{
  hipfftExecC2C(fft2_plan, d_in, d_out, HIPFFT_FORWARD);
  scaledata<<<gridsize, blocksize>>>(d_out, N, 1.f / sqrtf(rows*cols));
  arraydot<<<gridsize, blocksize>>>(d_out, d_out, d_mask, N);
}

void
backward (float2 *d_out, float2 *d_in, float2 *d_mask,
  const size_t N, const size_t rows, const size_t cols)
{
  float2 *d_tmp;
  cuTry(hipMalloc((void **)&d_tmp, N * sizeof(float2)));

  cuTry(hipMemcpyAsync(d_tmp, d_in, N * sizeof(float2),
          hipMemcpyDeviceToDevice, stream[0]));

  arraydot<<<gridsize, blocksize>>>(d_tmp, d_tmp, d_mask, N);
  hipfftExecC2C(fft2_plan, d_tmp, d_out, HIPFFT_BACKWARD);
  scaledata<<<gridsize, blocksize>>>(d_out, N, 1.f / sqrtf(rows*cols));

  cuTry(hipFree(d_tmp));
}

void
update_r (float2 *d_r, float2 *d_lbar, float2 *d_sbar,
  float2 *d_imgb, float2 *d_tmp, float2 *d_mask, float sigma,
  const size_t N, const size_t rows, const size_t cols)
{
  arrayadd<<<gridsize, blocksize>>>(d_tmp, d_lbar, d_sbar, N, 1.f, 1.f);
  forward(d_tmp, d_tmp, d_mask, N, rows, cols);
  update_r<<<gridsize, blocksize>>>(d_r, d_tmp, d_imgb, N, sigma);
}

__global__ void
update_r (float2 *d_r, float2 *d_tmp, float2 *d_imgb,
  size_t N, float sigma)
{
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N;
    id += blockDim.x * gridDim.x)
    {
      d_tmp[id] = d_tmp[id] - d_imgb[id];
      d_r[id] = (d_r[id] + sigma*d_tmp[id])/(1.f+sigma);
    }
}

__global__ void
grad_xx (float2 *d_out, float2 *d_in, const size_t N,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N - rows*ndyn;
    id += blockDim.x * gridDim.x)
  {
    // (the index inside a group) + (the index of the group) * offset
    idx = id%(rows*(cols-1)) + id/(rows*(cols-1))*(rows*cols);

    d_out[idx] = d_in[idx+rows] - d_in[idx];
  }
}

__global__ void
grad_xx_bound(float2 *d_array, const size_t N,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < rows*ndyn;
    id += blockDim.x * gridDim.x)
    {
      idx = id%rows + id/rows*(rows*cols) + rows*(cols-1);

      d_array[idx] = make_float2(0.f);
    }
}

__global__ void
grad_yy (float2 *d_out, float2 *d_in, const size_t N,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N - cols*ndyn;
    id += blockDim.x * gridDim.x)
  {
    idx = id%(rows-1) + id/(rows-1)*rows;

    d_out[idx] = d_in[idx+1] - d_in[idx];
  }
}

__global__ void
grad_yy_bound(float2 *d_array, const size_t N,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < cols*ndyn;
    id += blockDim.x * gridDim.x)
    {
      idx = id*rows + (rows-1);

      d_array[idx] = make_float2(0.f);
    }
}

__global__ void
grad_tt (float2 *d_out, float2 *d_in, const size_t N,
  const size_t rows, const size_t cols, const size_t ndyn,
  float mu)
{
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N - rows*cols;
    id += blockDim.x * gridDim.x)
  {
    d_out[id] = (d_in[id+rows*cols] - d_in[id])/mu;
  }
}

__global__ void
grad_tt_bound(float2 *d_array, const size_t N,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < rows*cols;
    id += blockDim.x * gridDim.x)
    {
      d_array[id + (ndyn-1)*rows*cols] = make_float2(0.f);
    }
}

void
grad (float2 *d_out, float2 *d_in, const size_t N,
    const size_t rows, const size_t cols, const size_t ndyn,
    float mu, char mode)
  {
    switch (mode)
    {
      case 'x':
        grad_xx<<<gridsize, blocksize>>>(d_out, d_in, N, rows, cols, ndyn);
        grad_xx_bound<<<gridsize, blocksize>>>(d_out, N, rows, cols, ndyn);
        break;
      case 'y':
        grad_yy<<<gridsize, blocksize>>>(d_out, d_in, N, rows, cols, ndyn);
        grad_yy_bound<<<gridsize, blocksize>>>(d_out, N, rows, cols, ndyn);
        break;
      case 't':
        grad_tt<<<gridsize, blocksize>>>(d_out, d_in, N, rows, cols, ndyn, mu);
        grad_tt_bound<<<gridsize, blocksize>>>(d_out, N, rows, cols, ndyn);
        break;
      default:
        printf("Please select x, y or z!\n");
        break;
    }
  }

__global__ void
proj_p (float2 *d_p, float2 *d_tmp, float2 *d_wbar,
   const size_t N, float sigma, float alpha)
{
  float absp, denom;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N;
    id += blockDim.x * gridDim.x)
  {
    d_p[id] = d_p[id] - sigma*(d_tmp[id] + d_wbar[id]);
    d_p[id+N] = d_p[id+N] - sigma*(d_tmp[id+N] + d_wbar[id+N]);
    d_p[id+2*N] = d_p[id+2*N] - sigma*(d_tmp[id+2*N] + d_wbar[id+2*N]);

    absp = sqrtf(abs(d_p[id])*abs(d_p[id]) + abs(d_p[id+N])*abs(d_p[id+N])
      + abs(d_p[id+2*N])*abs(d_p[id+2*N]));
    denom = fmaxf(1.f, absp/alpha);

    d_p[id] = d_p[id]/denom;
    d_p[id+N] = d_p[id+N]/denom;
    d_p[id+2*N] = d_p[id+2*N]/denom;
  }
}

__global__ void
grad_xx_adj (float2 *d_out, float2 *d_in, const size_t N,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N - 2*rows*ndyn;
    id += blockDim.x * gridDim.x)
    {
      idx = id%(rows*(cols-2)) + id/(rows*(cols-2))*(rows*cols) + rows;

      d_out[idx] = d_in[idx] - d_in[idx-rows];
    }
}

__global__ void
grad_xx_adj_zero (float2 *d_out, float2 *d_in,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < rows*ndyn;
    id += blockDim.x * gridDim.x)
    {
      idx = id%rows + id/rows*(rows*cols);

      d_out[idx] = d_in[idx];
    }
}

__global__ void
grad_xx_adj_bound (float2 *d_out, float2 *d_in,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < rows*ndyn;
    id += blockDim.x * gridDim.x)
    {
      idx = id%rows + id/rows*(rows*cols) + rows*(cols-1);

      d_out[idx] = -1.f*(d_in[idx-rows]);
    }
}

__global__ void
grad_yy_adj (float2 *d_out, float2 *d_in, const size_t N,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N - 2*cols*ndyn;
    id += blockDim.x * gridDim.x)
    {
      idx = id%(rows-2) + id/(rows-2)*rows + 1;

      d_out[idx] = d_in[idx] - d_in[idx-1];
    }
}

__global__ void
grad_yy_adj_zero (float2 *d_out, float2 *d_in,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < cols*ndyn;
    id += blockDim.x * gridDim.x)
    {
      idx = id*rows;

      d_out[idx] = d_in[idx];
    }
}

__global__ void
grad_yy_adj_bound (float2 *d_out, float2 *d_in,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < cols*ndyn;
    id += blockDim.x * gridDim.x)
    {
      idx = id*rows + (rows-1);

      d_out[idx] = -1.f*d_in[idx-1];
    }
}

__global__ void
grad_tt_adj (float2 *d_out, float2 *d_in, const size_t N,
  const size_t rows, const size_t cols, const size_t ndyn,
  float mu)
{
  int stride = rows*cols;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N - 2*rows*cols;
    id += blockDim.x * gridDim.x)
    {
      d_out[id+stride] = (d_in[id+stride] - d_in[id])/mu;
    }
}

__global__ void
grad_tt_adj_zero (float2 *d_out, float2 *d_in,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < rows*cols;
    id += blockDim.x * gridDim.x)
    {
      idx = id;
      d_out[idx] = d_in[idx];
    }
}

__global__ void
grad_tt_adj_bound (float2 *d_out, float2 *d_in,
  const size_t rows, const size_t cols, const size_t ndyn)
{
  int idx;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < rows*cols;
    id += blockDim.x * gridDim.x)
    {
      idx = id + rows*cols*(ndyn-1);
      d_out[idx] = -1.f*d_in[idx-rows*cols];
    }
}

void
grad_adj (float2 *d_out, float2 *d_in, const size_t N,
  const size_t rows, const size_t cols, const size_t ndyn,
  float mu, char mode)
{
  switch (mode)
  {
    case 'x':
      grad_xx_adj<<<gridsize, blocksize>>>(d_out, d_in, N, rows, cols, ndyn);
      grad_xx_adj_zero<<<gridsize, blocksize>>>(d_out, d_in, rows, cols, ndyn);
      grad_xx_adj_bound<<<gridsize, blocksize>>>(d_out, d_in, rows, cols, ndyn);
      break;
    case 'y':
      grad_yy_adj<<<gridsize, blocksize>>>(d_out, d_in, N, rows, cols, ndyn);
      grad_yy_adj_zero<<<gridsize, blocksize>>>(d_out, d_in, rows, cols, ndyn);
      grad_yy_adj_bound<<<gridsize, blocksize>>>(d_out, d_in, rows, cols, ndyn);
      break;
    case 't':
      grad_tt_adj<<<gridsize, blocksize>>>(d_out, d_in, N, rows, cols, ndyn, mu);
      grad_tt_adj_zero<<<gridsize, blocksize>>>(d_out, d_in, rows, cols, ndyn);
      grad_tt_adj_bound<<<gridsize, blocksize>>>(d_out, d_in, rows, cols, ndyn);
      break;
    default:
      printf("Please select x, y or z!\n");
      break;
  }
}

__global__ void
proj_q (float2 *d_q, float2 *d_tmp,
   const size_t N, float sigma, float alpha)
{
  float absq, denom;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N;
    id += blockDim.x * gridDim.x)
    {
      d_tmp[id] = d_tmp[id+3*N];
      d_tmp[id+N] = d_tmp[id+4*N];
      d_tmp[id+2*N] = d_tmp[id+5*N];
      d_tmp[id+3*N] = (d_tmp[id+6*N] + d_tmp[id+7*N])/2.f;
      d_tmp[id+4*N] = (d_tmp[id+8*N] + d_tmp[id+9*N])/2.f;
      d_tmp[id+5*N] = (d_tmp[id+10*N] + d_tmp[id+11*N])/2.f;

      d_q[id] = d_q[id] - sigma*d_tmp[id];
      d_q[id+N] = d_q[id+N] - sigma*d_tmp[id+N];
      d_q[id+2*N] = d_q[id+2*N] - sigma*d_tmp[id+2*N];
      d_q[id+3*N] = d_q[id+3*N] - sigma*d_tmp[id+3*N];
      d_q[id+4*N] = d_q[id+4*N] - sigma*d_tmp[id+4*N];
      d_q[id+5*N] = d_q[id+5*N] - sigma*d_tmp[id+5*N];

      absq = sqrtf(abs(d_q[id])*abs(d_q[id])
                 + abs(d_q[id+N])*abs(d_q[id+N])
                 + abs(d_q[id+2*N])*abs(d_q[id+2*N])
                 + 2*abs(d_q[id+3*N])*abs(d_q[id+3*N])
                 + 2*abs(d_q[id+4*N])*abs(d_q[id+4*N])
                 + 2*abs(d_q[id+5*N])*abs(d_q[id+5*N]));
      denom = fmaxf(1.f, absq/alpha);

      d_q[id] = d_q[id]/denom;
      d_q[id+N] = d_q[id+N]/denom;
      d_q[id+2*N] = d_q[id+2*N]/denom;
      d_q[id+3*N] = d_q[id+3*N]/denom;
      d_q[id+4*N] = d_q[id+4*N]/denom;
      d_q[id+5*N] = d_q[id+5*N]/denom;
    }
}

__global__ void
update_s (float2 *d_imgs, float2 *d_tmp, float2 *d_imgz,
  const size_t N, float tau)
{
  float2 divp;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N;
    id += blockDim.x * gridDim.x)
    {
      divp = d_tmp[id] + d_tmp[id+N] + d_tmp[id+2*N];
      d_imgs[id] = d_imgs[id] - tau*(d_imgz[id] + divp);
    }
}

__global__ void
update_w (float2 *d_w, float2 *d_tmp, float2 *d_p,
  const size_t N, float tau)
{
  float2 divq;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N;
    id += blockDim.x * gridDim.x)
    {
      divq = d_tmp[id+3*N] + d_tmp[id+4*N] + d_tmp[id+5*N];
      d_w[id] = d_w[id] - tau*(divq - d_p[id]);

      divq = d_tmp[id+6*N] + d_tmp[id+7*N] + d_tmp[id+8*N];
      d_w[id+N] = d_w[id+N] - tau*(divq - d_p[id+N]);

      divq = d_tmp[id+9*N] + d_tmp[id+10*N] + d_tmp[id+11*N];
      d_w[id+2*N] = d_w[id+2*N] - tau*(divq - d_p[id+2*N]);
    }
}

__global__ void
shrink(float2 *d_array2, float *d_array, const float beta, const int array_size)
{
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < array_size;
    id += blockDim.x * gridDim.x)
    {
      d_array[id] = fmaxf(d_array[id] - beta, 0.f);
      d_array2[id] = make_float2(d_array[id]);
    }
}

__global__ void
arrayabs(float *d_array, float2 *d_array2, const size_t array_size)
{
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < array_size;
    id += blockDim.x * gridDim.x)
    {
      d_array[id] = abs(d_array2[id]);
    }
}

float
compute_ser(float2 *d_array, float2 *d_img, const size_t array_size)
{
  float *d_diff, *d_tmp, ser, diff_norm, img_norm;

  cuTry(hipMalloc((void **)&d_diff, array_size * sizeof(float)));
  cuTry(hipMalloc((void **)&d_tmp, array_size * sizeof(float)));

  arrayabs<<<gridsize, blocksize>>>(d_diff, d_array, array_size);
  arrayabs<<<gridsize, blocksize>>>(d_tmp, d_img, array_size);
  arrayadd<<<gridsize, blocksize>>>(d_diff, d_diff, d_tmp, array_size, 1.f, -1.f);

  hipblas_status = hipblasSnrm2(hipblas_handle, array_size, d_tmp, 1, &img_norm);
  assert(hipblas_status == HIPBLAS_STATUS_SUCCESS);

  hipblas_status = hipblasSnrm2(hipblas_handle, array_size, d_diff, 1, &diff_norm);
  assert(hipblas_status == HIPBLAS_STATUS_SUCCESS);

  ser = -20.f*log10f(diff_norm/img_norm);

  cuTry(hipFree(d_diff));
  cuTry(hipFree(d_tmp));

  return ser;
}

void
tgv_cs(float2 *d_imgl, float2 *d_imgs, float2 *h_img, float2 *h_mask,
      const size_t N, const size_t rows, const size_t cols, const size_t ndyn,
      float alpha, float beta, float mu,
      float tau, float sigma, float reduction, int iter)
{
  // Setup cuda handles
  hipblas_status = hipblasCreate(&hipblas_handle);
  hipsolver_status = hipsolverDnCreate(&hipsolver_handle);

  // Read image and mask
  float2 *d_img, *d_imgz, *d_imgb, *d_mask;

  cuTry(hipMalloc((void **) &d_img, N * sizeof(float2)));
  cuTry(hipMemcpyAsync(d_img, h_img, N * sizeof(float2),
          hipMemcpyHostToDevice, stream[0]));

  cuTry(hipMalloc((void **) &d_mask, N * sizeof(float2)));
  cuTry(hipMemcpyAsync(d_mask, h_mask, N * sizeof(float2),
          hipMemcpyHostToDevice, stream[0]));

  // Scale the data to [0,1]
  float img_max = compute_maxmag(d_img, N);
  scaledata<<<gridsize, blocksize>>>(d_img, N, 1.f / img_max);

  // Sample image using mask and generate zerofilled image
  cuTry(hipMalloc((void **) &d_imgz, N * sizeof(float2)));
  cuTry(hipMalloc((void **) &d_imgb, N * sizeof(float2)));

  fft2_init(rows, cols, ndyn);

  forward(d_imgb, d_img, d_mask, N, rows, cols);
  backward(d_imgz, d_imgb, d_mask, N, rows, cols);

  // Allocate arrays
  float2 *d_w; // Primal variables
  float2 *d_p, *d_q, *d_r; // Dual variables
  float2 *d_lbar, *d_sbar, *d_wbar; // Intermediate variables
  float2 *d_lold, *d_sold, *d_wold; // Old variables
  float2 *d_tmp; // Temporal variable (grad, grad_adj ect)

  cuTry(hipMalloc((void **)&d_w, 3 * N * sizeof(float2)));
  cuTry(hipMalloc((void **)&d_p, 3 * N * sizeof(float2)));
  cuTry(hipMalloc((void **)&d_q, 6 * N *sizeof(float2)));
  cuTry(hipMalloc((void **)&d_r, N * sizeof(float2)));
  cuTry(hipMalloc((void **)&d_lbar, N * sizeof(float2)));
  cuTry(hipMalloc((void **)&d_sbar, N * sizeof(float2)));
  cuTry(hipMalloc((void **)&d_wbar, 3 * N * sizeof(float2)));
  cuTry(hipMalloc((void **)&d_lold, N * sizeof(float2)));
  cuTry(hipMalloc((void **)&d_sold, N * sizeof(float2)));
  cuTry(hipMalloc((void **)&d_wold, 3 * N * sizeof(float2)));
  cuTry(hipMalloc((void **)&d_tmp, 12 * N * sizeof(float2)));

  // Initialization for variables
  arrayreal<<<gridsize, blocksize>>>(d_imgl, d_imgz, N);
  cuTry(hipMemset(d_imgs, 0.f, N * sizeof(float2)));
  cuTry(hipMemset(d_w, 0.f, 3 * N * sizeof(float2)));

  cuTry(hipMemcpyAsync(d_sbar, d_imgs, N * sizeof(float2),
          hipMemcpyDeviceToDevice, stream[0]));
  cuTry(hipMemcpyAsync(d_lbar, d_imgl, N * sizeof(float2),
          hipMemcpyDeviceToDevice, stream[0]));
  cuTry(hipMemcpyAsync(d_wbar, d_w, 3 * N * sizeof(float2),
          hipMemcpyDeviceToDevice, stream[0]));

  cuTry(hipMemset(d_p, 0.f, 3 * N * sizeof(float2)));
  cuTry(hipMemset(d_q, 0.f, 6 * N * sizeof(float2)));
  cuTry(hipMemset(d_r, 0.f, N * sizeof(float2)));

  cuTry(hipMemset(d_tmp, 0.f, 12 * N * sizeof(float2)));

  // Initialization for alpha
  float alpha0 = sqrtf(2)*alpha;
  float alpha1 = alpha;

  float alpha00 = alpha0;
  float alpha10 = alpha1;
  float alpha01 = alpha0*reduction;
  float alpha11 = alpha1*reduction;

  // Initialization for SVD
  int lwork, lda, ldu, lds, ldvt, *devInfo;

  hipsolver_status = hipsolverDnCgesvd_bufferSize(hipsolver_handle,
      rows*cols, ndyn, &lwork);
  assert(hipsolver_status == HIPSOLVER_STATUS_SUCCESS);

  lda = rows*cols;
  ldu = rows*cols;
  lds = fminf(ndyn, rows*cols);
  ldvt = ndyn;

  cuTry(hipMalloc((void **)&devInfo, sizeof(int)));

  float2 *d_lu, *d_ls2, *d_lvt, *d_lsvt, *d_work, a, b;
  float *d_ls, *d_rwork;

  cuTry(hipMalloc((void **)&d_lu, ldu*lds*sizeof(float2)));
  cuTry(hipMalloc((void **)&d_ls, lds*sizeof(float)));
  cuTry(hipMalloc((void **)&d_ls2, lds*sizeof(float2)));
  cuTry(hipMalloc((void **)&d_lvt, lds*ldvt*sizeof(float2)));
  cuTry(hipMalloc((void **)&d_lsvt, lds*ldvt*sizeof(float2)));
  cuTry(hipMalloc((void **)&d_work, lwork*sizeof(float2)));
  cuTry(hipMalloc((void **)&d_rwork, (lds-1)*sizeof(float)));

  a = make_float2(1.f);
  b = make_float2(0.f);

  // Run the main loop
  printf("Running recon ...\n");
  for (int i = 1; i <= iter; i++)
  {
    // update alpha
    alpha0 = compute_alpha(alpha00, alpha01, iter, i);
    alpha1 = compute_alpha(alpha10, alpha11, iter, i);

    cuTry(hipMemcpyAsync(d_lold, d_imgl, N * sizeof(float2),
            hipMemcpyDeviceToDevice, stream[0]));
    cuTry(hipMemcpyAsync(d_sold, d_imgs, N * sizeof(float2),
            hipMemcpyDeviceToDevice, stream[0]));
    cuTry(hipMemcpyAsync(d_wold, d_w, 3 * N * sizeof(float2),
            hipMemcpyDeviceToDevice, stream[0]));

    // update r
    update_r(d_r, d_lbar, d_sbar, d_imgb, d_tmp, d_mask,
      sigma, N, rows, cols);

    // update p
    grad(d_tmp, d_sbar, N, rows, cols, ndyn, mu, 'x');
    grad(d_tmp+N, d_sbar, N, rows, cols, ndyn, mu, 'y');
    grad(d_tmp+2*N, d_sbar, N, rows, cols, ndyn, mu, 't');

    proj_p<<<gridsize, blocksize>>>(d_p, d_tmp, d_wbar, N, sigma, alpha1);

    // update q
    grad_adj(d_tmp+3*N, d_wbar, N, rows, cols, ndyn, mu, 'x');
    grad_adj(d_tmp+4*N, d_wbar+N, N, rows, cols, ndyn, mu, 'y');
    grad_adj(d_tmp+5*N, d_wbar+2*N, N, rows, cols, ndyn, mu, 't');
    grad_adj(d_tmp+6*N, d_wbar, N, rows, cols, ndyn, mu, 'y');
    grad_adj(d_tmp+7*N, d_wbar+N, N, rows, cols, ndyn, mu, 'x');
    grad_adj(d_tmp+8*N, d_wbar+2*N, N, rows, cols, ndyn, mu, 'x');
    grad_adj(d_tmp+9*N, d_wbar, N, rows, cols, ndyn, mu, 't');
    grad_adj(d_tmp+10*N, d_wbar+N, N, rows, cols, ndyn, mu, 't');
    grad_adj(d_tmp+11*N, d_wbar+2*N, N, rows, cols, ndyn, mu, 'y');

    proj_q<<<gridsize, blocksize>>>(d_q, d_tmp, N, sigma, alpha0);

    // update s
    backward(d_imgz, d_r, d_mask, N, rows, cols);

    grad_adj(d_tmp, d_p, N, rows, cols, ndyn, mu, 'x');
    grad_adj(d_tmp+N, d_p+N, N, rows, cols, ndyn, mu, 'y');
    grad_adj(d_tmp+2*N, d_p+2*N, N, rows, cols, ndyn, mu, 't');

    update_s<<<gridsize, blocksize>>>(d_imgs, d_tmp, d_imgz, N, tau);

    // update w
    grad(d_tmp+3*N, d_q, N, rows, cols, ndyn, mu, 'x');
    grad(d_tmp+4*N, d_q+3*N, N, rows, cols, ndyn, mu, 'y');
    grad(d_tmp+5*N, d_q+4*N, N, rows, cols, ndyn, mu, 't');
    grad(d_tmp+6*N, d_q+3*N, N, rows, cols, ndyn, mu, 'x');
    grad(d_tmp+7*N, d_q+N, N, rows, cols, ndyn, mu, 'y');
    grad(d_tmp+8*N, d_q+5*N, N, rows, cols, ndyn, mu, 't');
    grad(d_tmp+9*N, d_q+4*N, N, rows, cols, ndyn, mu, 'x');
    grad(d_tmp+10*N, d_q+5*N, N, rows, cols, ndyn, mu, 'y');
    grad(d_tmp+11*N, d_q+2*N, N, rows, cols, ndyn, mu, 't');

    update_w<<<gridsize, blocksize>>>(d_w, d_tmp, d_p, N, tau);

    // update l
    arrayadd<<<gridsize, blocksize>>>(d_imgl, d_imgl, d_imgz, N, 1.f, -1.f*tau);

    hipsolver_status = hipsolverDnCgesvd(hipsolver_handle, 'S', 'S',
      rows*cols, ndyn, d_imgl, lda, d_ls, d_lu, ldu, d_lvt, ldvt,
      d_work, lwork, d_rwork, devInfo);
    assert(hipsolver_status == HIPSOLVER_STATUS_SUCCESS);

    shrink<<<gridsize, blocksize>>>(d_ls2, d_ls, beta, lds);

    hipblas_status = hipblasCdgmm(hipblas_handle, HIPBLAS_SIDE_LEFT,
      lds, ldvt, d_lvt, ldvt, d_ls2, 1, d_lsvt, ldvt);
    assert(hipblas_status == HIPBLAS_STATUS_SUCCESS);

    hipblas_status = hipblasCgemm(hipblas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
      ldu, ldvt, lds, &a, d_lu, ldu, d_lsvt, ldvt, &b, d_imgl, ldu);
    assert(hipblas_status == HIPBLAS_STATUS_SUCCESS);

    // update intermediate variables
    arrayadd<<<gridsize, blocksize>>>(d_sbar, d_imgs, d_sold, N, 2.f, -1.f);
    arrayadd<<<gridsize, blocksize>>>(d_wbar, d_w, d_wold, 3*N, 2.f, -1.f);
    arrayadd<<<gridsize, blocksize>>>(d_lbar, d_imgl, d_lold, N, 2.f, -1.f);
  }

  // Compute SER
  float2 *d_imgr;
  cuTry(hipMalloc((void **)&d_imgr, N * sizeof(float2)));

  arrayadd<<<gridsize, blocksize>>>(d_imgr, d_imgl, d_imgs, N, 1.f, 1.f);
  backward(d_imgz, d_imgb, d_mask, N, rows, cols);

  float ser_imgz, ser_imgr;
  ser_imgz = compute_ser(d_imgz, d_img, N);
  ser_imgr = compute_ser(d_imgr, d_img, N);

  printf("The SER of zerofill: %.2f dB\n", ser_imgz);
  printf("The SER of recon:    %.2f dB\n", ser_imgr);

  // Free arrays and destroy handles
  cuTry(hipFree(d_img));
  cuTry(hipFree(d_imgb));
  cuTry(hipFree(d_imgz));
  cuTry(hipFree(d_imgr));
  cuTry(hipFree(d_mask));

  cuTry(hipFree(d_w));
  cuTry(hipFree(d_p));
  cuTry(hipFree(d_q));
  cuTry(hipFree(d_r));
  cuTry(hipFree(d_lbar));
  cuTry(hipFree(d_sbar));
  cuTry(hipFree(d_wbar));
  cuTry(hipFree(d_lold));
  cuTry(hipFree(d_sold));
  cuTry(hipFree(d_wold));
  cuTry(hipFree(d_tmp));

  cuTry(hipFree(d_lu));
  cuTry(hipFree(d_ls));
  cuTry(hipFree(d_ls2));
  cuTry(hipFree(d_lvt));
  cuTry(hipFree(d_lsvt));
  cuTry(hipFree(d_work));
  cuTry(hipFree(d_rwork));
  cuTry(hipFree(devInfo));

  hipblasDestroy(hipblas_handle);
  hipsolverDnDestroy(hipsolver_handle);
}
