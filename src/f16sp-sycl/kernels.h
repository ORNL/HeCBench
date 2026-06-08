#define NUM_OF_BLOCKS (1024 * 1024)
#define NUM_OF_THREADS 128

inline
void reduceInShared_native(sycl::half2 *const v, sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);
  #pragma unroll
  for (int i = NUM_OF_THREADS/2; i >= 1; i = i / 2) {
    if(lid<i) v[lid] = v[lid] + v[lid+i];
    item.barrier(sycl::access::fence_space::local_space);
  }
}

inline
void reduceInShared_native(sycl::float2 *const v, sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);
  #pragma unroll
  for (int i = NUM_OF_THREADS/2; i >= 1; i = i / 2) {
    if(lid<i) v[lid] = v[lid] + v[lid+i];
    item.barrier(sycl::access::fence_space::local_space);
  }
}


void scalarProductKernel_native(const sycl::half2 *a,
                                const sycl::half2 *b,
                                float *results,
                                      sycl::half2 *shArray,
                                const size_t size,
                                sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);

  const size_t stride = item.get_group_range(0) * item.get_local_range(0);

  sycl::half2 value(0.f, 0.f);
  shArray[lid] = value;

  for (size_t i = item.get_global_id(0); i < size; i += stride)
  {
    value = a[i] * b[i] + value;
  }

  shArray[lid] = value;
  item.barrier(sycl::access::fence_space::local_space);
  reduceInShared_native(shArray, item);

  if (lid == 0)
  {
    sycl::half2 result = shArray[0];
    float f_result = (float)result.y() + (float)result.x();
    auto ao = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                     sycl::memory_scope::device,\
                     sycl::access::address_space::global_space>(results[0]);
    ao.fetch_add(f_result);
  }
}

void scalarProductKernel_native_fp32(const sycl::half2 *a,
                                     const sycl::half2 *b,
                                     float *results,
                                           sycl::float2 *shArray,
                                     const size_t size,
                                     sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);

  const size_t stride = item.get_group_range(0) * item.get_local_range(0);

  sycl::float2 value(0.f, 0.f);
  shArray[lid] = value;

  for (size_t i = item.get_global_id(0); i < size; i += stride)
  {
    value.x() += (float)a[i].x() * (float)b[i].x();
    value.y() += (float)a[i].y() * (float)b[i].y();
  }

  shArray[lid] = value;
  item.barrier(sycl::access::fence_space::local_space);
  reduceInShared_native(shArray, item);

  if (lid == 0)
  {
    sycl::float2 result = shArray[0];
    auto ao = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                     sycl::memory_scope::device,\
                     sycl::access::address_space::global_space>(results[0]);
    ao.fetch_add(result.x() + result.y());
  }
}

void scalarProductKernel_native2_fp32(const sycl::half2 *a,
                                      const sycl::half2 *b,
                                      float *results,
                                      const size_t size,
                                      sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);

  const size_t stride = item.get_group_range(0) * item.get_local_range(0);

  sycl::float2 value(0.f, 0.f);

  for (size_t i = item.get_global_id(0); i < size; i += stride)
  {
    value.x() += (float)a[i].x() * (float)b[i].x();
    value.y() += (float)a[i].y() * (float)b[i].y();
  }

  value = sycl::reduce_over_group(item.get_group(), value, sycl::plus<sycl::float2>{});

  if (lid == 0)
  {
    auto ao = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                     sycl::memory_scope::device,\
                     sycl::access::address_space::global_space>(results[0]);
    ao.fetch_add(value.x() + value.y());
  }
}

