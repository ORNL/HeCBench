#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>

typedef float DTYPE;

void maxpool3d(
  const DTYPE* i_img,
        DTYPE* o_img,
  const int Hstride,
  const int Vstride,
  const int pool_width,
  const int pool_height,
  const int i_img_count,
  const int i_img_width,
  const int i_img_height,
  const int o_img_width,
  const int o_img_height,
  sycl::nd_item<3> &item )
{
  const int x = item.get_global_id(2);
  const int y = item.get_global_id(1);
  const int z = item.get_global_id(0);
  if (x >= o_img_width || y >= o_img_height || z >= i_img_count)
    return;

  const int xidx = Hstride*x;
  const int yidx = Vstride*y;
  DTYPE maxval = (DTYPE)0;

  for (int r = 0; r < pool_height; r++)
  {
    const int idxIntmp = ((z*i_img_height + yidx + r) * i_img_width) + xidx;
    for(int c = 0; c < pool_width; c++)
    {
      const int idxIn = idxIntmp + c;
      maxval = sycl::fmax(maxval, i_img[idxIn]);
    }
  }
  o_img[(((z * o_img_height) + y) * o_img_width) + x] = maxval;
}

int main(int argc, char** argv)
{
  if (argc != 5) {
    printf("Usage: %s <image width> <image height> <image count> <repeat>\n", argv[0]);
    return 1;
  }
  int i_img_width  = atoi(argv[1]);
  int i_img_height = atoi(argv[2]);
  int i_img_count = atoi(argv[3]);
  int repeat = atoi(argv[4]);

  int Hstride=2, Vstride=2;
  int o_img_width  = i_img_width/Hstride;
  int o_img_height = i_img_height/Vstride;

  printf("input image width %d Hstride %d\n", i_img_width,Hstride);
  printf("input image height %d Vstride %d\n", i_img_height,Vstride);
  printf("output image width %d\n", o_img_width);
  printf("output image height %d\n", o_img_height);

  // Generate random values for each image
  int size_image = i_img_width * i_img_height;
  size_t mem_size_image = sizeof(DTYPE) * size_image;
  DTYPE *h_image  = (DTYPE*)malloc(mem_size_image * i_img_count);

  srand(2);

  for(int j=0;j<i_img_count;j++) {
    for(int i=0;i<size_image;i++) {
      h_image[(j*size_image)+i] = rand()%256 / (DTYPE)255;
    }
  }

  // host and device results
  int size_output = o_img_width * o_img_height;
  size_t mem_size_output = sizeof(DTYPE) * size_output;
  DTYPE* h_output = (DTYPE*) malloc(mem_size_output*i_img_count);
  DTYPE* d_output = (DTYPE*) malloc(mem_size_output*i_img_count);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Create the input and output arrays in device memory
  DTYPE *d_image = sycl::malloc_device<DTYPE>(size_image*i_img_count, q);
  q.memcpy(d_image, h_image, mem_size_image*i_img_count);

  DTYPE *d_result = sycl::malloc_device<DTYPE>(size_output*i_img_count, q);

  sycl::range<3> lws (4, 8, 8);
  sycl::range<3> gws ((i_img_count+3)/4*4,
                      (o_img_height+7)/8*8,
                      (o_img_width+7)/8*8);

  // filter size same as stride size
  const int pool_width  = Hstride;
  const int pool_height = Vstride;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.submit([&] (sycl::handler &h) {
      h.parallel_for<class maxpool3>(
      sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
        maxpool3d(d_image, d_result, Hstride, Vstride,
                  pool_width, pool_height, i_img_count, i_img_width, i_img_height,
                  o_img_width, o_img_height, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(d_output, d_result, mem_size_output*i_img_count).wait();

  // verification using the CPU results
  for (int z = 0; z < i_img_count; z++) {
    for (int y = 0; y < o_img_height; y++) {
      for (int x = 0; x < o_img_width; x++) {
        const int xidx = Hstride*x;
        const int yidx = Vstride*y;
        DTYPE maxval = (DTYPE)0;
        for (int r = 0; r < pool_height; r++)
        {
          const int idxIntmp = ((z*i_img_height + yidx + r) * i_img_width) + xidx;
          for(int c = 0; c < pool_width; c++)
          {
            const int idxIn = idxIntmp + c;
            maxval = fmaxf(maxval, h_image[idxIn]);
          }
        }
        h_output[(((z*o_img_height)+y)*o_img_width)+x] = maxval;
      }
    }
  }

  int status = memcmp(h_output, d_output, sizeof(DTYPE)*i_img_count*o_img_width*o_img_height);
  printf("%s\n", (status == 0) ? "PASS" : "FAIL");
  if (!(status == 0)) exit(1);

  free(h_image);
  free(h_output);
  free(d_output);
  sycl::free(d_image, q);
  sycl::free(d_result, q);
  return status;
}
