#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "reference.h"
#include "common.h"

// GPU thread block size
#define TPB 1024

inline void gpu_check(hipError_t error, const char *file, int line) {
  if (error != hipSuccess) {
    printf("[HIP ERROR] at file %s:%d:\n%s\n", file, line,
           hipGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}
#define GPU_CHECK(err) (gpu_check(err, __FILE__, __LINE__))

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeat, Kernel kernel, KernelArgs&&... kernel_args) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    kernel(std::forward<KernelArgs>(kernel_args)...);
  }
  GPU_CHECK(hipDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-3f) / repeat;
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference,
                     const char* name, std::uint64_t num_elements,
                     T tolerance=1e-4, int n_print=5, int check_all=0)
{
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    GPU_CHECK(hipMemcpy(out_gpu, device_result, num_elements * sizeof(D), hipMemcpyDeviceToHost));
    int nfaults = 0;
    for (uint64_t i = 0; i < num_elements; i++) {
      if (std::fabs(cpu_reference[i] - (T)out_gpu[i]) > tolerance && std::isfinite(cpu_reference[i])) {
        printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
        nfaults++;
        if (nfaults >= max_int(10, n_print)) {
          free(out_gpu);
        }
      }
    }

    free(out_gpu);
}

// -----------------------------------------------------------------------------------------------
// GPU kernels

template <int WarpSize>
__device__ __forceinline__
float cg_reduce (cooperative_groups::thread_block_tile<WarpSize,
                                                       cooperative_groups::thread_block> &warp,
                 float sum)
{
  for (int offset = WarpSize/2; offset > 0; offset /= 2) {
    sum += warp.shfl_xor(sum, offset);
  }
  return sum;
}

// Essentially taken from llm.c's kernel 5
// using kernel 5 because for images, each "channel" is effectively
// H * W * group_size, which is quite large
// One block per group of group_size channels: B * C / group_size (B * n_groups) blocks
__global__ void groupnorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int B, int C, int img_size, int group_size, int n_groups
) {
#if defined(__GFX8__) || defined(__GFX9__)
  #define WarpSize 64
#else
  #define WarpSize 32
#endif
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    __shared__ float shared_sum[TPB/WarpSize];
    __shared__ float shared_sum2[TPB/WarpSize];
    int num_warps = blockDim.x / WarpSize;
    int warp_id = threadIdx.x / WarpSize;
    int lane_id = threadIdx.x % WarpSize;

    // pixels in each group
    int group_pixels = img_size * group_size;

    // group index for each batch ranges from 0 to n_groups-1
    int g = blockIdx.x % n_groups;

    // move pointers
    x += blockIdx.x * group_pixels;
    out += blockIdx.x * group_pixels;
    // each block acceses group_size channels
    weight += g * group_size;
    bias += g * group_size;

    float thread_sum = 0.0f;
    float thread_sum2 = 0.0f;

    for (int i = threadIdx.x; i < group_pixels; i += blockDim.x) {
        float val = x[i];
        thread_sum += val;
        thread_sum2 += val * val;
    }

    // reduce from all warps
    float warp_sum = cg_reduce<WarpSize>(warp, thread_sum);
    float warp_sum2 = cg_reduce<WarpSize>(warp, thread_sum2);
    // store warp sum into shared memory
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();

    // load warp sums from shared memory
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    float block_sum = cg_reduce<WarpSize>(warp, warp_sum);
    float block_sum2 = cg_reduce<WarpSize>(warp, warp_sum2);

    block_sum /= group_pixels;
    block_sum2 /= group_pixels;
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = rsqrtf(var + 1e-5f);
    if (mean != nullptr && threadIdx.x == 0) {
        mean[blockIdx.x] = m;
    }
    if (rstd != nullptr && threadIdx.x == 0) {
        rstd[blockIdx.x] = s;
    }

    for (int i = threadIdx.x; i < group_pixels; i += blockDim.x) {
        int c = i / img_size;
        float n = s * (x[i] - m);
        out[i] = n * weight[c] + bias[c];
    }
}

__global__ void groupnorm_forward_kernel2(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int B, int C, int img_size, int group_size, int n_groups
) {
#if defined(__GFX8__) || defined(__GFX9__)
  #define WarpSize 64
#else
  #define WarpSize 32
#endif
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    __shared__ float shared_sum[TPB/WarpSize];
    __shared__ float shared_sum2[TPB/WarpSize];
    int num_warps = blockDim.x / WarpSize;
    int warp_id = threadIdx.x / WarpSize;
    int lane_id = threadIdx.x % WarpSize;

    // pixels in each group
    int group_pixels = img_size * group_size;

    // group index for each batch ranges from 0 to n_groups-1
    int g = blockIdx.x % n_groups;

    // move pointers
    x += blockIdx.x * group_pixels;
    out += blockIdx.x * group_pixels;
    // each block acceses group_size channels
    weight += g * group_size;
    bias += g * group_size;

    // vectorize the accesses
    int vec_count = group_pixels / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);

    float4 thread_sum_v4 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 thread_sum2_v4 = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 v = x_vec[i];
      thread_sum_v4.x  += v.x;
      thread_sum_v4.y  += v.y;
      thread_sum_v4.z  += v.z;
      thread_sum_v4.w  += v.w;
      thread_sum2_v4.x += v.x * v.x;
      thread_sum2_v4.y += v.y * v.y;
      thread_sum2_v4.z += v.z * v.z;
      thread_sum2_v4.w += v.w * v.w;
    }

    float thread_sum = thread_sum_v4.x + thread_sum_v4.y + thread_sum_v4.z + thread_sum_v4.w;
    float thread_sum2 = thread_sum2_v4.x + thread_sum2_v4.y + thread_sum2_v4.z + thread_sum2_v4.w;

    for (int i = vec_count * 4 + threadIdx.x; i < group_pixels; i += blockDim.x) {
        float val = x[i];
        thread_sum  += val;
        thread_sum2 += val * val;
    }

    // reduce from all warps
    float warp_sum = cg_reduce<WarpSize>(warp, thread_sum);
    float warp_sum2 = cg_reduce<WarpSize>(warp, thread_sum2);
    // store warp sum into shared memory
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();

    // load warp sums from shared memory
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    float block_sum = cg_reduce<WarpSize>(warp, warp_sum);
    float block_sum2 = cg_reduce<WarpSize>(warp, warp_sum2);

    block_sum /= group_pixels;
    block_sum2 /= group_pixels;
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = rsqrtf(var + 1e-5f);
    if (mean != nullptr && threadIdx.x == 0) {
        mean[blockIdx.x] = m;
    }
    if (rstd != nullptr && threadIdx.x == 0) {
        rstd[blockIdx.x] = s;
    }

    for (int i = threadIdx.x; i < group_pixels; i += blockDim.x) {
        int c = i / img_size;
        float n = s * (x[i] - m);
        out[i] = n * weight[c] + bias[c];
    }
}

// most similar to kernel 2 in llm.c
// not doing any float16 optimizations yet
// main change is to have each block allocated to a single group
__global__ void groupnorm_backward_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ weight,
    float* __restrict__ dx,
    float* __restrict__ dweight,
    float* __restrict__ dbias,
    int B, int C, int img_size, int group_size, int n_groups
) {
#if defined(__GFX8__) || defined(__GFX9__)
  #define WarpSize 64
#else
  #define WarpSize 32
#endif
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    __shared__ float shared_sum[TPB/WarpSize];
    __shared__ float shared_sum2[TPB/WarpSize];
    int num_warps = blockDim.x / WarpSize;
    int warp_id = threadIdx.x / WarpSize;
    int lane_id = threadIdx.x % WarpSize;
    int group_pixels = img_size * group_size;
    // group index
    int g = blockIdx.x % n_groups;

    // move pointers
    dout += blockIdx.x * img_size * group_size;
    x += blockIdx.x * img_size * group_size;
    dx += blockIdx.x * img_size * group_size;
    weight += g * group_size;
    dweight += g * group_size;
    dbias += g * group_size;

    float m_val = mean[blockIdx.x];
    float rstd_val = rstd[blockIdx.x];

    // calculate the two mean terms in the group dimension
    // first is dout * weight, and second is dout * weight * norm
    // where norm = (x - mean) * rstd
    float w_dout_thread = 0.0f;
    float w_dout_norm_thread = 0.0f;
    for (int i = threadIdx.x; i < group_pixels; i += blockDim.x) {
        int c = i / img_size;
        float cur_w_dout = weight[c] * dout[i];
        w_dout_thread += cur_w_dout;
        float norm = (x[i] - m_val) * rstd_val;
        w_dout_norm_thread += cur_w_dout * norm;
    }
    // warp reduce
    float w_dout_warp = cg_reduce<WarpSize>(warp, w_dout_thread);
    float w_dout_norm_warp = cg_reduce<WarpSize>(warp, w_dout_norm_thread);
    // store warp sum in shared mem
    shared_sum[warp_id] = w_dout_warp;
    shared_sum2[warp_id] = w_dout_norm_warp;
    __syncthreads();

    // load warp sums from shared memory
    w_dout_warp = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    w_dout_norm_warp = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    float w_dout_block = cg_reduce<WarpSize>(warp, w_dout_warp);
    float w_dout_norm_block = cg_reduce<WarpSize>(warp, w_dout_norm_warp);
    w_dout_block /= group_pixels;
    w_dout_norm_block /= group_pixels;

    // update dx
    for (int i = threadIdx.x; i < group_pixels; i += blockDim.x) {
        // in bounds of image
        // accumulate dw and db
        float dout_val = dout[i];
        float norm = (x[i] - m_val) * rstd_val;

        // update dx
        int c = (i / img_size) % group_size;
        float w_dout = weight[c] * dout_val;
        dx[i] = (w_dout - w_dout_block - norm * w_dout_norm_block) * rstd_val;
    }
    // update dw and db
    // use different methods when the image size is large or small

    // if the image size is larger than the block size
    // loop over the channels and use the whole block on each channel
    // otherwise, assign each warp to a channel
    // in either case image size must be larger than the warp size
    //assert(img_size % warp.size() == 0);
    //assert(blockDim.x % warp.size() == 0);
    if (img_size % blockDim.x == 0) {
        for (int c = 0; c < group_size; c++) {
            float dw_thread = 0.0f;
            float db_thread = 0.0f;
            for (int i = threadIdx.x; i < img_size; i += blockDim.x) {
                float dout_val = dout[i];
                db_thread += dout_val;
                float norm = (x[i] - m_val) * rstd_val;
                dw_thread += dout_val * norm;
            }

            // move pointers
            dout += img_size;
            x += img_size;

            // warp reduce
            float dw_warp = cg_reduce<WarpSize>(warp, dw_thread);
            float db_warp = cg_reduce<WarpSize>(warp, db_thread);
            ////// store warp sum in shared mem
            if (lane_id == 0) {
                shared_sum[warp_id] = dw_warp;
                shared_sum2[warp_id] = db_warp;
            }
            __syncthreads();
            // use the first thread to reduce the shared memory sums and save to global memory
            if (threadIdx.x == 0) {
                float dw_block = 0.0f;
                float db_block = 0.0f;
                for (int i = 0; i < num_warps; i++) {
                    dw_block += shared_sum[i];
                    db_block += shared_sum2[i];
                }
                atomicAdd(dweight + c, dw_block);
                atomicAdd(dbias + c, db_block);
            }
        }
    } else {
        // if group size is large, need to loop over the group channels with the whole block
        int block_reps = ceil_div(group_size, num_warps);
        for (int br = 0; br < block_reps; br++) {
            float dw_thread = 0.0f;
            float db_thread = 0.0f;

            int ch = br * num_warps + warp_id;
            if (ch < group_size) {
                const float* dout_ch = dout + ch * img_size;
                const float* x_ch = x + ch * img_size;
                for (int i = lane_id; i < img_size; i += warp.size()) {
                    float dout_val = dout_ch[i];
                    db_thread += dout_val;
                    float norm = (x_ch[i] - m_val) * rstd_val;
                    dw_thread += dout_val * norm;
                }

                // warp reduce
                float dw_warp = cg_reduce<WarpSize>(warp, dw_thread);
                float db_warp = cg_reduce<WarpSize>(warp, db_thread);
                // since each warp takes care of an entire image
                // directly store result
                if (lane_id == 0) {
                    atomicAdd(dweight + ch, dw_warp);
                    atomicAdd(dbias + ch, db_warp);
                }
            }
        }
    }
}

// kernel launcher

void groupnorm_forward(
    const float* x, const float* weight, const float* bias,
    float* out, float* mean, float* rstd,
    int B, int C, int H, int W, int n_groups
) {
    int img_size = H * W;
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(TPB, img_size * group_size), 32);
    groupnorm_forward_kernel<<<n_blocks, block_size>>>(
        x, weight, bias, out, mean, rstd, B, C, img_size, group_size, n_groups
    );
}

void groupnorm_forward2(
    const float* x, const float* weight, const float* bias,
    float* out, float* mean, float* rstd,
    int B, int C, int H, int W, int n_groups
) {
    int img_size = H * W;
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(TPB, img_size * group_size), 32);
    groupnorm_forward_kernel2<<<n_blocks, block_size>>>(
        x, weight, bias, out, mean, rstd, B, C, img_size, group_size, n_groups
    );
}

void groupnorm_backward(
    const float* dout, const float* x, const float* mean, const float* rstd, const float* weight,
    float* dx, float* dweight, float* dbias,
    int B, int C, int H, int W, int n_groups
) {
    int img_size = H * W;
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(TPB, img_size * group_size), 32 * group_size);
    groupnorm_backward_kernel<<<n_blocks, block_size>>>(
        dout, x, mean, rstd, weight, dx, dweight, dbias, B, C, img_size, group_size, n_groups
    );
}


// -----------------------------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc != 7) {
      printf("Usage: %s <batch size> <number of channels> <height> <width> <number of groups> <repeat>\n", argv[0]);
      return 1;
    }

    uint64_t B = atoi(argv[1]);
    uint64_t C = atoi(argv[2]);
    uint64_t H = atoi(argv[3]);
    uint64_t W = atoi(argv[4]);
    uint64_t n_groups = atoi(argv[5]);
    int repeat = atoi(argv[6]);

    uint64_t img_size = H * W;

    srand(0);
    float *x = make_random_float(B * C * img_size);
    float *weight = make_random_float(C);
    float *bias = make_random_float(C);
    float *dout = make_random_float(B * C * img_size);

    float *out = (float*)malloc(B * C * img_size * sizeof(float));

    // reset values
    float *dx = (float*)calloc(B * C * img_size , sizeof(float));
    float *dweight = (float*)calloc(C , sizeof(float));
    float *dbias = (float*)calloc(C , sizeof(float));

    float *mean = (float*) malloc(B * n_groups * sizeof(float));
    float *rstd = (float*) malloc(B * n_groups * sizeof(float));

    // device memory
    float *d_x, *d_weight, *d_bias, *d_out, *d_mean, *d_rstd, *d_dout, *d_dx, *d_dweight, *d_dbias;
    GPU_CHECK(hipMalloc(&d_x, B * C * img_size * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_weight, C * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_bias, C * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_out, B * C * img_size * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_mean, B * n_groups * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_rstd, B * n_groups * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_dout, B * C * img_size * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_dx, B * C * img_size * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_dweight, C * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_dbias, C * sizeof(float)));
    GPU_CHECK(hipMemcpy(d_x, x, B * C * img_size * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_weight, weight, C * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_bias, bias, C * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_dout, dout, B * C * img_size * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_dx, 0, B * C * img_size * sizeof(float)));
    GPU_CHECK(hipMemset(d_dweight, 0, C * sizeof(float)));
    GPU_CHECK(hipMemset(d_dbias, 0, C * sizeof(float)));

    printf("Checking forward pass\n");

    groupnorm_forward_ref(x, weight, bias, out, mean, rstd, B, C, img_size, n_groups);

    float fwd_acc = 1e-2;
    groupnorm_forward(d_x, d_weight, d_bias, d_out, d_mean, d_rstd, B, C, H, W, n_groups);
    validate_result(d_out, out, "out", B * C * img_size, fwd_acc);
    printf("Forward pass successful\n");

    printf("Checking forward2 pass\n");

    groupnorm_forward2(d_x, d_weight, d_bias, d_out, d_mean, d_rstd, B, C, H, W, n_groups);
    validate_result(d_out, out, "out", B * C * img_size, fwd_acc);
    printf("Forward2 pass successful\n");

    printf("Checking backward pass\n");

    groupnorm_backward_ref(dout, x, mean, rstd, weight,
                           dx, dweight, dbias,
                           B, C, img_size, n_groups);

    groupnorm_backward(d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias, B, C, H, W, n_groups);
    float acc = 1e-2;
    printf("Checking dbias\n");
    validate_result(d_dbias, dbias, "dbias", C, acc);
    printf("Checking dweight\n");
    validate_result(d_dweight, dweight, "dweight", C, acc);
    printf("Checking dx\n");
    validate_result(d_dx, dx, "dx", B * C * img_size, 1.0f);
    printf("Backward pass successful\n");
    printf("\n─────────────────────────────────────────────────────\n");

    printf("All results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmarks\n");
    float elapsed_time = benchmark_kernel(repeat, groupnorm_forward,
                                          d_x, d_weight, d_bias, d_out, d_mean, d_rstd,
                                          B, C, H, W, n_groups);
    printf("time %.4f us\n", elapsed_time);

    printf("Forward2 pass benchmarks\n");
    elapsed_time = benchmark_kernel(repeat, groupnorm_forward2,
                                    d_x, d_weight, d_bias, d_out, d_mean, d_rstd,
                                    B, C, H, W, n_groups);
    printf("time %.4f us\n", elapsed_time);

    printf("Backward pass benchmarks\n");
    elapsed_time = benchmark_kernel(repeat, groupnorm_backward,
                                    d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias,
                                    B, C, H, W, n_groups);
    printf("time %.4f us\n", elapsed_time);

    GPU_CHECK(hipFree(d_x));
    GPU_CHECK(hipFree(d_weight));
    GPU_CHECK(hipFree(d_bias));
    GPU_CHECK(hipFree(d_out));
    GPU_CHECK(hipFree(d_mean));
    GPU_CHECK(hipFree(d_rstd));
    GPU_CHECK(hipFree(d_dout));
    GPU_CHECK(hipFree(d_dx));
    GPU_CHECK(hipFree(d_dweight));
    GPU_CHECK(hipFree(d_dbias));

    free(x);
    free(weight);
    free(bias);
    free(out);
    free(dout);
    free(dx);
    free(dweight);
    free(dbias);
    free(mean);
    free(rstd);

    return 0;
}
