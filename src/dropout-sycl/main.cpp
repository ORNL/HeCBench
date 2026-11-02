#include <cstdio>
#include <chrono>
#include <utility> // std::pair
#ifdef USE_ONEMATH
#include <oneapi/math.hpp>
#include <oneapi/math/rng/device.hpp>
namespace math_ns = oneapi::math;
#else
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
namespace math_ns = oneapi::mkl;
#endif
#include <sycl/sycl.hpp>

// philox generates 128 bits of randomness at a time.
// Kernel uses this explicitly by putting suitably transformed result into float4
// for all members of float4 to be consumed UNROLL has to be 4. Don't change!
const int UNROLL = 4;

template <typename scalar_t, typename accscalar_t, typename IndexType,
          typename mask_t>
void fused_dropout_kernel(const scalar_t *__restrict__ a,
                          scalar_t *__restrict__ b, mask_t *__restrict__ c,
                          IndexType totalElements, accscalar_t p,
                          std::pair<uint64_t, uint64_t> seeds,
                          const sycl::nd_item<1> &item)
{
  accscalar_t scale = accscalar_t(1)/p;
  IndexType idx = item.get_global_id(0);
  IndexType blockDim = item.get_local_range(0);
  IndexType gridDim = item.get_group_range(0);

  math_ns::rng::device::philox4x32x10<4> engine (
      seeds.first, {seeds.second, idx * 4});

  IndexType rounded_size = ((totalElements - 1) / (blockDim * gridDim * UNROLL) + 1) *
                           blockDim * gridDim * UNROLL;

  for (IndexType linearIndex = idx; linearIndex < rounded_size;
       linearIndex += gridDim * blockDim * UNROLL) {

    math_ns::rng::device::uniform<float> distr;
    sycl::float4 rand = math_ns::rng::device::generate(distr, engine);

    scalar_t src[UNROLL];
    rand.x() = rand.x() < p;
    rand.y() = rand.y() < p;
    rand.z() = rand.z() < p;
    rand.w() = rand.w() < p;

    #pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim * gridDim * ii;
      if (li < totalElements) {
        const IndexType aOffset = li;
        src[ii] = a[aOffset];
      }
    }

    #pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim * gridDim * ii;
      if (li < totalElements) {
        const IndexType bOffset = li;
        b[bOffset] = src[ii] * (&rand.x())[ii] * scale;
        c[bOffset] = (mask_t)(&rand.x())[ii];
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
}

// aligned vector generates vectorized load/store
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <typename scalar_t, typename accscalar_t, typename IndexType,
          typename mask_t, int VEC>
void fused_dropout_kernel_vec(const scalar_t *__restrict__ a,
                              scalar_t *__restrict__ b, mask_t *__restrict__ c,
                              IndexType totalElements, accscalar_t p,
                              std::pair<uint64_t, uint64_t> seeds,
                              const sycl::nd_item<1> &item)
{
  using LoadT = aligned_vector<scalar_t, VEC>;
  using MaskLoadT = aligned_vector<mask_t, VEC>;

  // Helps align the total number of times curand_uniform4 is called by each thread for the same totalElements
  bool gridxvec_loop_state = 0;

  accscalar_t scale = accscalar_t(1)/p;
  IndexType idx = item.get_global_id(0);
  IndexType blockDim = item.get_local_range(0);
  IndexType gridDim = item.get_group_range(0);

  math_ns::rng::device::philox4x32x10<4> engine (
      seeds.first, {seeds.second, idx * 4});

  // Note: Vectorized loads means we'll stride each thread by an additional VEC factor, as we'll load VEC elements at a time
  for (IndexType linearIndex = idx * VEC; linearIndex < totalElements;
       linearIndex += gridDim * blockDim * VEC) {

    scalar_t src[VEC];
    LoadT *value = reinterpret_cast<LoadT*>(&src);

    sycl::float4 rand;
    if ((VEC == 4) || (gridxvec_loop_state == 0)) {
      math_ns::rng::device::uniform<float> distr;
      rand = math_ns::rng::device::generate(distr, engine);
    } else {
      // sets up the last two values we generated last iteration to be used this iteration.
      rand.x() = rand.z();
      rand.y() = rand.w();
      gridxvec_loop_state ^= 1;
    }
    rand.x() = rand.x() < p;
    rand.y() = rand.y() < p;
    if (VEC == 4) {
      rand.z() = rand.z() < p;
      rand.w() = rand.w() < p;
    }

    *value = *reinterpret_cast<const LoadT*>(&a[linearIndex]);

    scalar_t r[VEC];
    mask_t mask[VEC];

    // Perform the actual computation
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      r[ii] = src[ii] * (&rand.x())[ii] * scale;
      mask[ii] = (mask_t)(&rand.x())[ii];
    }
    // Vectorized writes for both mask & result
    *(reinterpret_cast<LoadT*>(&b[linearIndex])) = *reinterpret_cast<LoadT*>(&r[0]);
    *(reinterpret_cast<MaskLoadT*>(&c[linearIndex])) = *reinterpret_cast<MaskLoadT*>(&mask[0]);

    item.barrier(sycl::access::fence_space::local_space);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  const int64_t nelem = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  const int64_t block_size = 256;

  sycl::range<1> dim_block(block_size);
  sycl::range<1> grid_vec1(512);
  sycl::range<1> grid_vec2(grid_vec1[0] / 2);
  sycl::range<1> grid_vec4(grid_vec1[0] / 4);

  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  rng_engine_inputs.first =  12345678;
  rng_engine_inputs.second = 87654321;

  int64_t self_size = nelem * sizeof(float);
  int64_t ret_size = self_size;
  int64_t mask_size = nelem * sizeof(uint8_t);

  float *self_info = (float*) malloc (self_size);
  float *ret_info = (float*) malloc (ret_size);
  uint8_t *mask_info = (uint8_t*) malloc (mask_size);

  for (int64_t i = 0; i < nelem; i++) {
    self_info[i] = 0.1f;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_self_info = (float *)sycl::malloc_device(self_size, q);
  q.memcpy(d_self_info, self_info, self_size).wait();

  float *d_ret_info = (float *)sycl::malloc_device(ret_size, q);

  uint8_t *d_mask_info = (uint8_t *)sycl::malloc_device(mask_size, q);

  auto start = std::chrono::steady_clock::now();
  for (int p = 1; p <= repeat; p++) {
    float pa = (float)p / repeat;
    q.parallel_for(
        sycl::nd_range<1>(grid_vec1 * dim_block, dim_block),
        [=](sycl::nd_item<1> item) {
          fused_dropout_kernel<float, float, unsigned int, uint8_t>(
              d_self_info, d_ret_info, d_mask_info, nelem, pa,
              rng_engine_inputs, item);
        });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (VEC1) %lf (s)\n", time * 1e-9f);

  start = std::chrono::steady_clock::now();
  for (int p = 1; p <= repeat; p++) {
    float pa = (float)p / repeat;
    q.parallel_for(
        sycl::nd_range<1>(grid_vec2 * dim_block, dim_block),
        [=](sycl::nd_item<1> item) {
          fused_dropout_kernel_vec<float, float, unsigned int, uint8_t, 2>(
              d_self_info, d_ret_info, d_mask_info, nelem, pa,
              rng_engine_inputs, item);
        });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (VEC2) %lf (s)\n", time * 1e-9f);

  start = std::chrono::steady_clock::now();
  for (int p = 1; p <= repeat; p++) {
    float pa = (float)p / repeat;
    q.parallel_for(
        sycl::nd_range<1>(grid_vec4 * dim_block, dim_block),
        [=](sycl::nd_item<1> item) {
          fused_dropout_kernel_vec<float, float, unsigned int, uint8_t, 4>(
              d_self_info, d_ret_info, d_mask_info, nelem, pa,
              rng_engine_inputs, item);
        });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (VEC4) %lf (s)\n", time * 1e-9f);

  sycl::free(d_self_info, q);
  sycl::free(d_ret_info, q);
  sycl::free(d_mask_info, q);
  free(self_info);
  free(ret_info);
  free(mask_info);

  return 0;
}
