/* This code is provided as supplementary material for the book
   chapter "Exploiting graphics processing units for computational
   biology and bioinformatics," by Payne, Sinnott-Armstrong, and
   Moore, to appear in "The Handbook of Research on Computational and
   Systems Biology: Interdisciplinary applications," by IGI Global.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <sycl/sycl.hpp>

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif

#define INSTANCES 224   /* # of instances */
#define ATTRIBUTES 4096 /* # of attributes */
#define THREADS 128    /* # of threads per block */

/* CPU implementation */
void CPU(int * data, int * distance) {
  /* compare all pairs of instances, accessing the attributes in
     row-major order */
#pragma omp parallel for collapse(2)
  for (int i = 0; i < INSTANCES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      for (int k = 0; k < ATTRIBUTES; k++) {
        distance[i + INSTANCES * j] += 
          (data[i * ATTRIBUTES + k] != data[j * ATTRIBUTES + k]);
      }
    }
  }
}

int main(int argc, char **argv) {

  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }
  
  const int iterations = atoi(argv[1]);

  /* host data */
  int *data; 
  char *data_char;
  int *cpu_distance; 
  int *gpu_distance; 

  /* used to time CPU and GPU implementations */
  double start_cpu, stop_cpu;
  double start_gpu, stop_gpu;
  float elapsedTime; 
  struct timeval tp;
  struct timezone tzp;
  /* verification result */ 
  int status;

  /* seed RNG */
  srand(2);

  /* allocate host memory */
  const size_t distance_bytes = INSTANCES * INSTANCES * sizeof(int);
  const size_t int32_data_bytes = INSTANCES * ATTRIBUTES * sizeof(int);
  const size_t int8_data_bytes = INSTANCES * ATTRIBUTES * sizeof(char);

  data = (int *)malloc(int32_data_bytes);
  data_char = (char *)malloc(int8_data_bytes);
  cpu_distance = (int *)malloc(distance_bytes);
  gpu_distance = (int *)malloc(distance_bytes);

  /* randomly initialize host data */
#pragma omp parallel for collapse(2)
  for (int i = 0; i < ATTRIBUTES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      data[i + ATTRIBUTES * j] = data_char[i + ATTRIBUTES * j] = random() % 3;
    }
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  /* allocate GPU memory */
  char *d_data = sycl::malloc_device<char>(INSTANCES * ATTRIBUTES, q);
  q.memcpy(d_data, data_char, int8_data_bytes);

  int *d_distance = sycl::malloc_device<int>(INSTANCES * INSTANCES, q);

  sycl::range<2> gws (INSTANCES, INSTANCES*THREADS);
  sycl::range<2> lws (1, THREADS);

  /* CPU */
  bzero(cpu_distance,INSTANCES*INSTANCES*sizeof(int));
  gettimeofday(&tp, &tzp);
  start_cpu = tp.tv_sec*1000000+tp.tv_usec;
  CPU(data, cpu_distance);
  gettimeofday(&tp, &tzp);
  stop_cpu = tp.tv_sec*1000000+tp.tv_usec;
  elapsedTime = stop_cpu - start_cpu;
  printf("CPU time: %f (us)\n",elapsedTime);

  elapsedTime = 0; 
  for (int n = 0; n < iterations; n++) {
    /* register GPU kernel */
    q.memset(d_distance, 0, distance_bytes).wait();

    gettimeofday(&tp, &tzp);
    start_gpu = tp.tv_sec*1000000+tp.tv_usec;

    q.submit([&] (sycl::handler &h) {
      h.parallel_for<class k1>(
        sycl::nd_range<2> (gws, lws), [=] (sycl::nd_item<2> item) {
        int idx = item.get_local_id(1);
        int gx = item.get_group(1);
        int gy = item.get_group(0);

        for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
          sycl::char4 j = ldg((sycl::char4 *)(d_data + i + ATTRIBUTES*gx));
          sycl::char4 k = ldg((sycl::char4 *)(d_data + i + ATTRIBUTES*gy));

        /* use a local variable (stored in register) to hold intermediate
           values. This reduces writes to global memory */
          int count = 0;

          if(j.x() ^ k.x()) 
            count++; 
          if(j.y() ^ k.y())
            count++;
          if(j.z() ^ k.z())
            count++;
          if(j.w() ^ k.w())
            count++;

          /* atomic write to global memory */
          auto ao = sycl::atomic_ref<int, 
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space> (d_distance[INSTANCES*gx + gy]);
           
          ao.fetch_add(count);
        }
      });
    }).wait();

    gettimeofday(&tp, &tzp);
    stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
    elapsedTime += stop_gpu - start_gpu;
  }

  q.memcpy(gpu_distance, d_distance, distance_bytes).wait();

  printf("Average kernel execution time: %f (us)\n", elapsedTime / iterations);
  status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  if (status != 0) printf("FAIL\n");
  else printf("PASS\n");
  if (status != 0) exit(1);

  elapsedTime = 0; 
  for (int n = 0; n < iterations; n++) {
    /* shared memory GPU kernel */
    q.memset(d_distance, 0, distance_bytes).wait();

    gettimeofday(&tp, &tzp);
    start_gpu = tp.tv_sec*1000000+tp.tv_usec;

    /*  coalesced GPU implementation of the all-pairs kernel using
        character data types, registers, and shared memory */
    q.submit([&] (sycl::handler &h) {
      sycl::local_accessor<int, 1> dist(sycl::range<1>(THREADS), h); 
      h.parallel_for<class k2>(
        sycl::nd_range<2> (gws, lws), [=] (sycl::nd_item<2> item) {
        int idx = item.get_local_id(1);
        int gx = item.get_group(1);
        int gy = item.get_group(0);

        /* each thread initializes its own location of the shared array */ 
        dist[idx] = 0;

        /* At this point, the threads must be synchronized to ensure that
           the shared array is fully initialized. */
        item.barrier(sycl::access::fence_space::local_space);

        for(int i = idx*4; i < ATTRIBUTES; i+=THREADS*4) {
          sycl::char4 j = ldg((sycl::char4 *)(d_data + i + ATTRIBUTES*gx));
          sycl::char4 k = ldg((sycl::char4 *)(d_data + i + ATTRIBUTES*gy));

        /* use a local variable (stored in register) to hold intermediate
           values. This reduces writes to global memory */
          char count = 0;
          if(j.x() ^ k.x()) 
            count++; 
          if(j.y() ^ k.y())
            count++;
          if(j.z() ^ k.z())
            count++;
          if(j.w() ^ k.w())
            count++;

          /* Increment shared array */
          dist[idx] += count;
        }

        /* Synchronize threads to make sure all have completed their updates
           of the shared array. Since the distances for each thread are read
           by thread 0 below, this must be ensured. Above, it was not
           necessary because each thread was accessing its own memory
           */
        item.barrier(sycl::access::fence_space::local_space);

        /* Perform balanced tree reduction across the shared memory */
        for (int stride = THREADS/2; stride > 0; stride /= 2) {
          if (idx < stride) {
            dist[idx] += dist[idx + stride];
          }
          item.barrier(sycl::access::fence_space::local_space);
        }

        if(idx == 0) {
          /* Thread 0 will then write the output to global memory. Note that
             this does not need to be performed atomically, because only one
             thread per block is writing to global memory, and each block
             corresponds to a unique memory address. 
             */
          d_distance[INSTANCES*gy + gx] = dist[0];
        }
      });
    }).wait();

    gettimeofday(&tp, &tzp);
    stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
    elapsedTime += stop_gpu - start_gpu;
  }

  q.memcpy(gpu_distance, d_distance, distance_bytes).wait();

  printf("Average kernel execution time: %f (us)\n", elapsedTime / iterations);
  status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  if (status != 0) printf("FAIL\n");
  else printf("PASS\n");
  if (status != 0) exit(1);

  elapsedTime = 0; 
  for (int n = 0; n < iterations; n++) {
    /* shared memory GPU kernel */
    q.memset(d_distance, 0, distance_bytes).wait();

    gettimeofday(&tp, &tzp);
    start_gpu = tp.tv_sec*1000000+tp.tv_usec;

    q.submit([&] (sycl::handler &h) {
      h.parallel_for<class k3>(
        sycl::nd_range<2> (gws, lws), [=] (sycl::nd_item<2> item) {
        int idx = item.get_local_id(1);
        int gx = item.get_group(1);
        int gy = item.get_group(0);

        int dist = 0;

        for(int i = idx*4; i < ATTRIBUTES; i+=THREADS*4) {
          sycl::char4 j = ldg((sycl::char4 *)(d_data + i + ATTRIBUTES*gx));
          sycl::char4 k = ldg((sycl::char4 *)(d_data + i + ATTRIBUTES*gy));

          char count = 0;
          if(j.x() ^ k.x()) 
            count++; 
          if(j.y() ^ k.y())
            count++;
          if(j.z() ^ k.z())
            count++;
          if(j.w() ^ k.w())
            count++;

          /* Increment shared array */
          dist += count;
        }

        int sum = sycl::reduce_over_group(item.get_group(), dist, std::plus<>());
        
        if(idx == 0) {
          d_distance[INSTANCES*gy + gx] = sum;
        }
      });
    }).wait();

    gettimeofday(&tp, &tzp);
    stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
    elapsedTime += stop_gpu - start_gpu;
  }

  q.memcpy(gpu_distance, d_distance, distance_bytes).wait();

  printf("Average kernel execution time: %f (us)\n", elapsedTime / iterations);
  status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  if (status != 0) printf("FAIL\n");
  else printf("PASS\n");
  if (status != 0) exit(1);

  free(cpu_distance);
  free(gpu_distance);
  free(data);
  free(data_char);
  sycl::free(d_data, q);
  sycl::free(d_distance, q);

  return status;
}
