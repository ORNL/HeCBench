#define NUM_OF_BLOCKS (1024 * 1024)
#define NUM_OF_THREADS 128
// __launch_bounds__ tells the compiler the block size for register allocation

// fp16 reduction
__forceinline__ __device__
void reduceInShared_intrinsics(half2 * const v)
{
  int lid = threadIdx.x;
  #pragma unroll
  for (int i = NUM_OF_THREADS/2; i >= 1; i = i / 2) {
    if(lid<i) v[lid] = __hadd2(v[lid], v[lid+i]);
    __syncthreads();
  }
}

// fp32 reduction
__forceinline__ __device__
void reduceInShared_native(float2 * const v)
{
  int lid = threadIdx.x;
  #pragma unroll
  for (int i = NUM_OF_THREADS/2; i >= 1; i = i / 2) {
    if(lid<i) {
      v[lid].x += v[lid+i].x;
      v[lid].y += v[lid+i].y;
    }
    __syncthreads();
  }
}


__launch_bounds__(NUM_OF_THREADS)
__global__
void scalarProductKernel_intrinsics(
    half2 const *__restrict__ const a,
    half2 const *__restrict__ const b,
    float *__restrict__ const results,
    size_t const size)
{
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  half2 value = __float2half2_rn(0.f);

  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
              i < size; i+=stride)
  {
    value = __hfma2(a[i], b[i], value);
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_intrinsics(shArray);

  if (threadIdx.x == 0)
  {
    half2 result = shArray[0];
    float f_result = __low2float(result) + __high2float(result);
    unsafeAtomicAdd(results, f_result);
  }
}

// compute type is float for high precision
__launch_bounds__(NUM_OF_THREADS)
__global__
void scalarProductKernel_native_fp32(
    half2 const *__restrict__ const a,
    half2 const *__restrict__ const b,
    float *__restrict__ const results,
    size_t const size)
{
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  __shared__ float2 shArray[NUM_OF_THREADS];

  float2 value = {0.f, 0.f};

  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
              i < size; i+=stride)
  {
    value.x += (float)a[i].x * (float)b[i].x;
    value.y += (float)a[i].y * (float)b[i].y;
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_native(shArray);

  if (threadIdx.x == 0)
  {
    float2 result = shArray[0];
    unsafeAtomicAdd(results, result.x + result.y);
  }
}


// CUB reduction with float2
__launch_bounds__(NUM_OF_THREADS)
__global__
void scalarProductKernel_native2_fp32(
    half2 const *__restrict__ const a,
    half2 const *__restrict__ const b,
    float *__restrict__ const results,
    size_t const size)
{
  const size_t stride = (size_t)gridDim.x * blockDim.x;

  typedef hipcub::BlockReduce<float2, NUM_OF_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  float2 value = {0.f, 0.f};

  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
              i < size; i+=stride)
  {
    value.x += (float)a[i].x * (float)b[i].x;
    value.y += (float)a[i].y * (float)b[i].y;
  }

  value = BlockReduce(temp_storage).Sum(value);

  if (threadIdx.x == 0)
  {
    unsafeAtomicAdd(results, value.x + value.y);
  }
}
