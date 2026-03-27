#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include "reference.h"
#include "common.h"

// GPU thread block size
#define TPB 1024

template<class Kernel, class... KernelArgs>
float benchmark_kernel(sycl::queue &q, int repeat, Kernel kernel, KernelArgs&&... kernel_args) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    kernel(std::forward<KernelArgs>(kernel_args)...);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-3f) / repeat;
}

template<class D, class T>
void validate_result(sycl::queue &q,
                     D* device_result, const T* cpu_reference,
                     const char* name, std::uint64_t num_elements,
                     T tolerance=1e-4, int n_print=5, int check_all=0)
{
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    q.memcpy(out_gpu, device_result, num_elements * sizeof(D)).wait();
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

// Essentially taken from llm.c's kernel 5
// using kernel 5 because for images, each "channel" is effectively
// H * W * group_size, which is quite large
// One block per group of group_size channels: B * C / group_size (B * n_groups) blocks
template <int WarpSize>
void groupnorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int B, int C, int img_size, int group_size, int n_groups,
    float *shared_sum,
    float *shared_sum2,
    sycl::nd_item<3> item)
{
    auto warp = item.get_sub_group();

    int num_warps = item.get_local_range(2) / WarpSize;
    int warp_id = item.get_local_id(2) / WarpSize;
    int lane_id = item.get_local_id(2) % WarpSize;

    // pixels in each group
    int group_pixels = img_size * group_size;

    // group index for each batch ranges from 0 to n_groups-1
    int g = item.get_group(2) % n_groups;

    // move pointers
    x += item.get_group(2) * group_pixels;
    out += item.get_group(2) * group_pixels;
    // each block acceses group_size channels
    weight += g * group_size;
    bias += g * group_size;

    float thread_sum = 0.0f;
    float thread_sum2 = 0.0f;

    for (int i = item.get_local_id(2); i < group_pixels;
         i += item.get_local_range(2)) {
        float val = x[i];
        thread_sum += val;
        thread_sum2 += val * val;
    }

    // reduce from all warps
    float warp_sum = sycl::reduce_over_group(warp, thread_sum, sycl::plus<float>{});
    float warp_sum2 = sycl::reduce_over_group(warp, thread_sum2, sycl::plus<float>{});
    // store warp sum into shared memory
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    item.barrier(sycl::access::fence_space::local_space);

    // load warp sums from shared memory
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    float block_sum = sycl::reduce_over_group(warp, warp_sum, sycl::plus<float>{});
    float block_sum2 = sycl::reduce_over_group(warp, warp_sum2, sycl::plus<float>{});

    block_sum /= group_pixels;
    block_sum2 /= group_pixels;
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = sycl::rsqrt(var + 1e-5f);
    if (mean != nullptr && item.get_local_id(2) == 0) {
        mean[item.get_group(2)] = m;
    }
    if (rstd != nullptr && item.get_local_id(2) == 0) {
        rstd[item.get_group(2)] = s;
    }

    for (int i = item.get_local_id(2); i < group_pixels;
         i += item.get_local_range(2)) {
        int c = i / img_size;
        float n = s * (x[i] - m);
        out[i] = n * weight[c] + bias[c];
    }
}

template <int WarpSize>
void groupnorm_forward_kernel2(const float *__restrict__ x,
                               const float *__restrict__ weight,
                               const float *__restrict__ bias,
                               float *__restrict__ out,
                               float *__restrict__ mean,
                               float *__restrict__ rstd, int B, int C,
                               int img_size, int group_size, int n_groups,
                               float *shared_sum, float *shared_sum2,
    sycl::nd_item<3> item)
{
    auto warp = item.get_sub_group();

    int num_warps = item.get_local_range(2) / WarpSize;
    int warp_id = item.get_local_id(2) / WarpSize;
    int lane_id = item.get_local_id(2) % WarpSize;

    // pixels in each group
    int group_pixels = img_size * group_size;

    // group index for each batch ranges from 0 to n_groups-1
    int g = item.get_group(2) % n_groups;

    // move pointers
    x += item.get_group(2) * group_pixels;
    out += item.get_group(2) * group_pixels;
    // each block acceses group_size channels
    weight += g * group_size;
    bias += g * group_size;

    // vectorize the accesses
    int vec_count = group_pixels / 4;
    const sycl::float4 *x_vec = reinterpret_cast<const sycl::float4 *>(x);

    sycl::float4 thread_sum_v4 = sycl::float4(0.f, 0.f, 0.f, 0.f);
    sycl::float4 thread_sum2_v4 = sycl::float4(0.f, 0.f, 0.f, 0.f);

    for (int i = item.get_local_id(2); i < vec_count;
         i += item.get_local_range(2)) {
      sycl::float4 v = x_vec[i];
      thread_sum_v4 += v;
      thread_sum2_v4 += v * v;
    }

    float thread_sum = thread_sum_v4.x() + thread_sum_v4.y() +
                       thread_sum_v4.z() + thread_sum_v4.w();
    float thread_sum2 = thread_sum2_v4.x() + thread_sum2_v4.y() +
                        thread_sum2_v4.z() + thread_sum2_v4.w();

    for (int i = vec_count * 4 + item.get_local_id(2); i < group_pixels;
         i += item.get_local_range(2)) {
        float val = x[i];
        thread_sum  += val;
        thread_sum2 += val * val;
    }

    // reduce from all warps
    float warp_sum = sycl::reduce_over_group(warp, thread_sum, sycl::plus<float>{});
    float warp_sum2 = sycl::reduce_over_group(warp, thread_sum2, sycl::plus<float>{});
    // store warp sum into shared memory
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    item.barrier(sycl::access::fence_space::local_space);

    // load warp sums from shared memory
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    float block_sum = sycl::reduce_over_group(warp, warp_sum, sycl::plus<float>{});
    float block_sum2 = sycl::reduce_over_group(warp, warp_sum2, sycl::plus<float>{});

    block_sum /= group_pixels;
    block_sum2 /= group_pixels;
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = sycl::rsqrt(var + 1e-5f);
    if (mean != nullptr && item.get_local_id(2) == 0) {
        mean[item.get_group(2)] = m;
    }
    if (rstd != nullptr && item.get_local_id(2) == 0) {
        rstd[item.get_group(2)] = s;
    }

    for (int i = item.get_local_id(2); i < group_pixels;
         i += item.get_local_range(2)) {
        int c = i / img_size;
        float n = s * (x[i] - m);
        out[i] = n * weight[c] + bias[c];
    }
}

// most similar to kernel 2 in llm.c
// not doing any float16 optimizations yet
// main change is to have each block allocated to a single group
template <int WarpSize>
void groupnorm_backward_kernel(
    const float *__restrict__ dout, const float *__restrict__ x,
    const float *__restrict__ mean, const float *__restrict__ rstd,
    const float *__restrict__ weight, float *__restrict__ dx,
    float *__restrict__ dweight, float *__restrict__ dbias, int B, int C,
    int img_size, int group_size, int n_groups, float *shared_sum,
    float *shared_sum2,
    sycl::nd_item<3> item)
{
    auto warp = item.get_sub_group();

    int num_warps = item.get_local_range(2) / WarpSize;
    int warp_id = item.get_local_id(2) / WarpSize;
    int lane_id = item.get_local_id(2) % WarpSize;
    int group_pixels = img_size * group_size;
    // group index
    int g = item.get_group(2) % n_groups;

    // move pointers
    dout += item.get_group(2) * img_size * group_size;
    x += item.get_group(2) * img_size * group_size;
    dx += item.get_group(2) * img_size * group_size;
    weight += g * group_size;
    dweight += g * group_size;
    dbias += g * group_size;

    float m_val = mean[item.get_group(2)];
    float rstd_val = rstd[item.get_group(2)];

    // calculate the two mean terms in the group dimension
    // first is dout * weight, and second is dout * weight * norm
    // where norm = (x - mean) * rstd
    float w_dout_thread = 0.0f;
    float w_dout_norm_thread = 0.0f;
    for (int i = item.get_local_id(2); i < group_pixels;
         i += item.get_local_range(2)) {
        int c = i / img_size;
        float cur_w_dout = weight[c] * dout[i];
        w_dout_thread += cur_w_dout;
        float norm = (x[i] - m_val) * rstd_val;
        w_dout_norm_thread += cur_w_dout * norm;
    }
    // warp reduce
    float w_dout_warp = sycl::reduce_over_group(warp, w_dout_thread, sycl::plus<float>{});
    float w_dout_norm_warp = sycl::reduce_over_group(warp, w_dout_norm_thread, sycl::plus<float>{});
    // store warp sum in shared mem
    shared_sum[warp_id] = w_dout_warp;
    shared_sum2[warp_id] = w_dout_norm_warp;
    item.barrier(sycl::access::fence_space::local_space);

    // load warp sums from shared memory
    w_dout_warp = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    w_dout_norm_warp = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    float w_dout_block = sycl::reduce_over_group(warp, w_dout_warp, sycl::plus<float>{});
    float w_dout_norm_block = sycl::reduce_over_group(warp, w_dout_norm_warp, sycl::plus<float>{});
    w_dout_block /= group_pixels;
    w_dout_norm_block /= group_pixels;

    // update dx
    for (int i = item.get_local_id(2); i < group_pixels;
         i += item.get_local_range(2)) {
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
    //assert(item.get_local_range(2) % WarpSize == 0);
    if (img_size % item.get_local_range(2) == 0) {
        for (int c = 0; c < group_size; c++) {
            float dw_thread = 0.0f;
            float db_thread = 0.0f;
            for (int i = item.get_local_id(2); i < img_size;
                 i += item.get_local_range(2)) {
                float dout_val = dout[i];
                db_thread += dout_val;
                float norm = (x[i] - m_val) * rstd_val;
                dw_thread += dout_val * norm;
            }

            // move pointers
            dout += img_size;
            x += img_size;

            // warp reduce
            float dw_warp = sycl::reduce_over_group(warp, dw_thread, sycl::plus<float>{});
            float db_warp = sycl::reduce_over_group(warp, db_thread, sycl::plus<float>{});
            ////// store warp sum in shared mem
            if (lane_id == 0) {
                shared_sum[warp_id] = dw_warp;
                shared_sum2[warp_id] = db_warp;
            }
            item.barrier(sycl::access::fence_space::local_space);
            // use the first thread to reduce the shared memory sums and save to global memory
            if (item.get_local_id(2) == 0) {
                float dw_block = 0.0f;
                float db_block = 0.0f;
                for (int i = 0; i < num_warps; i++) {
                    dw_block += shared_sum[i];
                    db_block += shared_sum2[i];
                }
                auto a_dw = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                                 sycl::memory_scope::device,\
                                 sycl::access::address_space::generic_space>(dweight[c]);
                a_dw.fetch_add(dw_block);

                auto a_db = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                                 sycl::memory_scope::device,\
                                 sycl::access::address_space::generic_space>(dbias[c]);
                a_db.fetch_add(db_block);
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
                for (int i = lane_id; i < img_size;
                     i += warp.get_local_linear_range()) {
                    float dout_val = dout_ch[i];
                    db_thread += dout_val;
                    float norm = (x_ch[i] - m_val) * rstd_val;
                    dw_thread += dout_val * norm;
                }

                // warp reduce
                float dw_warp = sycl::reduce_over_group(warp, dw_thread, sycl::plus<float>{});
                float db_warp = sycl::reduce_over_group(warp, db_thread, sycl::plus<float>{});
                // since each warp takes care of an entire image
                // directly store result
                if (lane_id == 0) {
                    auto a_dw = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                                     sycl::memory_scope::device,\
                                     sycl::access::address_space::generic_space>(dweight[ch]);
                    a_dw.fetch_add(dw_warp);

                    auto a_db = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                                     sycl::memory_scope::device,\
                                     sycl::access::address_space::generic_space>(dbias[ch]);
                    a_db.fetch_add(db_warp);
                }
            }
        }
    }
}

// kernel launcher

template <int WarpSize>
void groupnorm_forward(sycl::queue &q,
    const float* x, const float* weight, const float* bias,
    float* out, float* mean, float* rstd,
    int B, int C, int H, int W, int n_groups
) {
    int img_size = H * W;
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(TPB, img_size * group_size), WarpSize);
    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_sum_acc(sycl::range<1>(TPB/WarpSize), cgh);
        sycl::local_accessor<float, 1> shared_sum2_acc(sycl::range<1>(TPB/WarpSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, n_blocks * block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                groupnorm_forward_kernel<WarpSize>(
                    x, weight, bias, out, mean, rstd, B, C, img_size,
                    group_size, n_groups,
                    shared_sum_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_sum2_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    item);
            });
    });
}

template <int WarpSize>
void groupnorm_forward2(sycl::queue &q,
    const float* x, const float* weight, const float* bias,
    float* out, float* mean, float* rstd,
    int B, int C, int H, int W, int n_groups
) {
    int img_size = H * W;
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(TPB, img_size * group_size), WarpSize);
    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_sum_acc(sycl::range<1>(TPB/WarpSize), cgh);
        sycl::local_accessor<float, 1> shared_sum2_acc(sycl::range<1>(TPB/WarpSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, n_blocks * block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                groupnorm_forward_kernel2<WarpSize>(
                    x, weight, bias, out, mean, rstd, B, C, img_size,
                    group_size, n_groups,
                    shared_sum_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_sum2_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    item);
            });
    });
}

template <int WarpSize>
void groupnorm_backward(sycl::queue &q,
    const float* dout, const float* x, const float* mean, const float* rstd, const float* weight,
    float* dx, float* dweight, float* dbias,
    int B, int C, int H, int W, int n_groups
) {
    int img_size = H * W;
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(TPB, img_size * group_size), WarpSize * group_size);
    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_sum_acc(sycl::range<1>(TPB/WarpSize), cgh);
        sycl::local_accessor<float, 1> shared_sum2_acc(sycl::range<1>(TPB/WarpSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, n_blocks * block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                groupnorm_backward_kernel<WarpSize>(
                    dout, x, mean, rstd, weight, dx, dweight, dbias, B, C,
                    img_size, group_size, n_groups,
                    shared_sum_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_sum2_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    item);
            });
    });
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

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
    int warpSize = *r;

    // device memory
    float *d_x, *d_weight, *d_bias, *d_out, *d_mean, *d_rstd, *d_dout, *d_dx, *d_dweight, *d_dbias;
    d_x = sycl::malloc_device<float>(B * C * img_size, q);
    d_weight = sycl::malloc_device<float>(C, q);
    d_bias = sycl::malloc_device<float>(C, q);
    d_out = sycl::malloc_device<float>(B * C * img_size, q);
    d_mean = sycl::malloc_device<float>(B * n_groups, q);
    d_rstd = sycl::malloc_device<float>(B * n_groups, q);
    d_dout = sycl::malloc_device<float>(B * C * img_size, q);
    d_dx = sycl::malloc_device<float>(B * C * img_size, q);
    d_dweight = sycl::malloc_device<float>(C, q);
    d_dbias = sycl::malloc_device<float>(C, q);
    q.memcpy(d_x, x, B * C * img_size * sizeof(float));
    q.memcpy(d_weight, weight, C * sizeof(float));
    q.memcpy(d_bias, bias, C * sizeof(float));
    q.memcpy(d_dout, dout, B * C * img_size * sizeof(float));
    q.memset(d_dx, 0, B * C * img_size * sizeof(float));
    q.memset(d_dweight, 0, C * sizeof(float));
    q.memset(d_dbias, 0, C * sizeof(float));

    printf("Checking forward pass\n");

    groupnorm_forward_ref(x, weight, bias, out, mean, rstd, B, C, img_size, n_groups);

    float fwd_acc = 1e-2;
    if (warpSize == 64)
        groupnorm_forward<64>(q, d_x, d_weight, d_bias, d_out, d_mean, d_rstd, B, C, H, W, n_groups);
    else
        groupnorm_forward<32>(q, d_x, d_weight, d_bias, d_out, d_mean, d_rstd, B, C, H, W, n_groups);
    validate_result(q, d_out, out, "out", B * C * img_size, fwd_acc);
    printf("Forward pass successful\n");

    printf("Checking forward2 pass\n");

    if (warpSize == 64)
        groupnorm_forward2<64>(q, d_x, d_weight, d_bias, d_out, d_mean, d_rstd, B, C, H, W, n_groups);
    else
        groupnorm_forward2<32>(q, d_x, d_weight, d_bias, d_out, d_mean, d_rstd, B, C, H, W, n_groups);
    validate_result(q, d_out, out, "out", B * C * img_size, fwd_acc);
    printf("Forward2 pass successful\n");

    printf("Checking backward pass\n");

    groupnorm_backward_ref(dout, x, mean, rstd, weight, dx, dweight, dbias, B, C, img_size, n_groups);

    if (warpSize == 64)
        groupnorm_backward<64>(q, d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias, B, C, H, W, n_groups);
    else
        groupnorm_backward<32>(q, d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias, B, C, H, W, n_groups);
    float acc = 1e-2;
    printf("Checking dbias\n");
    validate_result(q, d_dbias, dbias, "dbias", C, acc);
    printf("Checking dweight\n");
    validate_result(q, d_dweight, dweight, "dweight", C, acc);
    printf("Checking dx\n");
    validate_result(q, d_dx, dx, "dx", B * C * img_size, 1.0f);
    printf("Backward pass successful\n");
    printf("\n─────────────────────────────────────────────────────\n");

    printf("All results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmarks\n");
    float elapsed_time;
    if (warpSize == 64)
        elapsed_time = benchmark_kernel(q, repeat, groupnorm_forward<64>, q,
                                          d_x, d_weight, d_bias, d_out, d_mean, d_rstd,
                                          B, C, H, W, n_groups);
    else
        elapsed_time = benchmark_kernel(q, repeat, groupnorm_forward<32>, q,
                                          d_x, d_weight, d_bias, d_out, d_mean, d_rstd,
                                          B, C, H, W, n_groups);
    printf("time %.4f us\n", elapsed_time);

    printf("Forward2 pass benchmarks\n");
    if (warpSize == 64)
        elapsed_time = benchmark_kernel(q, repeat, groupnorm_forward2<64>, q,
                                    d_x, d_weight, d_bias, d_out, d_mean, d_rstd,
                                    B, C, H, W, n_groups);
    else
        elapsed_time = benchmark_kernel(q, repeat, groupnorm_forward2<32>, q,
                                    d_x, d_weight, d_bias, d_out, d_mean, d_rstd,
                                    B, C, H, W, n_groups);
    printf("time %.4f us\n", elapsed_time);

    printf("Backward pass benchmarks\n");
    if (warpSize == 64)
        elapsed_time = benchmark_kernel(q, repeat, groupnorm_backward<64>, q,
                                    d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias,
                                    B, C, H, W, n_groups);
    else
        elapsed_time = benchmark_kernel(q, repeat, groupnorm_backward<32>, q,
                                    d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias,
                                    B, C, H, W, n_groups);
    printf("time %.4f us\n", elapsed_time);

    sycl::free(d_x, q);
    sycl::free(d_weight, q);
    sycl::free(d_bias, q);
    sycl::free(d_out, q);
    sycl::free(d_mean, q);
    sycl::free(d_rstd, q);
    sycl::free(d_dout, q);
    sycl::free(d_dx, q);
    sycl::free(d_dweight, q);
    sycl::free(d_dbias, q);

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
