#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "utils.h"
#include "reference.h"

void car (
  sycl::nd_item<1> &item,
  const float *__restrict img,
  const float *__restrict kernels,
  const float *__restrict offsets_h,
  const float *__restrict offsets_v,
        float *__restrict output,
  const params p,
  const int offset_unit,
  const int padding,
  const size_t n)
{
  size_t global_idx = item.get_global_id(0);
  if(global_idx >= n) return;

  const int dim_b = p.output_dim_b;
  const int dim_c = p.output_dim_c;
  const int dim_h = p.output_dim_h;
  const int dim_w = p.output_dim_w;
  const int kernels_size = p.kernel_size;
  const int img_w = p.image_w;
  const int img_h = p.image_h;

  const size_t vol_size = (size_t)dim_c * dim_h * dim_w;
  const size_t img_size = (size_t)dim_h * dim_w;

  const int idb = (global_idx / vol_size) % dim_b;
  const int idc = (global_idx / img_size) % dim_c;
  const int idy = (global_idx / dim_w) % dim_h;
  const int idx = global_idx % dim_w;

  const int k_size = (int)sycl::sqrt(float(kernels_size));
  const int w = img_w - 2 * padding;
  const int h = img_h - 2 * padding;

  float result = 0;
  for(int k_y = 0; k_y < k_size; ++k_y)
  {
    for(int k_x = 0; k_x < k_size; ++k_x)
    {
      const float offset_h = offsets_h(idb,k_size * k_y + k_x,idy,idx) * offset_unit;
      const float offset_v = offsets_v(idb,k_size * k_y + k_x,idy,idx) * offset_unit;

      const float p_x = static_cast<float>(idx + 0.5f) / dim_w * w + k_x + offset_h - 0.5f;
      const float p_y = static_cast<float>(idy + 0.5f) / dim_h * h + k_y + offset_v - 0.5f;
      const float alpha = p_x - sycl::floor(p_x);
      const float beta = p_y - sycl::floor(p_y);

      const int xL = sycl::max(sycl::min(int(sycl::floor(p_x)), w + 2 * padding - 1), 0);
      const int xR = sycl::max(sycl::min(xL + 1, w + 2 * padding - 1), 0);
      const int yT = sycl::max(sycl::min(int(sycl::floor(p_y)), h + 2 * padding - 1), 0);
      const int yB = sycl::max(sycl::min(yT + 1, h + 2 * padding - 1), 0);

      float val = (1.f - alpha) * (1.f - beta) * img(idb,idc,yT,xL);
      val += alpha * (1.f - beta) * img(idb,idc,yT,xR);
      val += (1.f - alpha) * beta * img(idb,idc,yB,xL);
      val += alpha * beta * img(idb,idc,yB,xR);
      result += val * kernels(idb,k_size * k_y + k_x,idy,idx);
    }
  }
  output(idb,idc,idy,idx) = result;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  params p = {128, 3, 480, 640, 9, 1024, 1024};
  const int dim_b = p.output_dim_b;
  const int dim_c = p.output_dim_c;
  const int dim_h = p.output_dim_h;
  const int dim_w = p.output_dim_w;
  const int kernels_size = p.kernel_size;
  const int img_w = p.image_w;
  const int img_h = p.image_h;

  const int padding = 1;

  size_t image_size = (size_t)dim_b * dim_c * (img_w + padding) * (img_h + padding);
  size_t offset_size = (size_t)dim_b * kernels_size * dim_w * dim_h;
  size_t kernel_size = (size_t)dim_b * kernels_size * dim_w * dim_h;
  size_t output_size = (size_t)dim_b * dim_c * dim_w * dim_h;

  size_t image_size_byte = sizeof(float) * image_size;
  size_t offset_size_byte = sizeof(float) * offset_size;
  size_t kernel_size_byte = sizeof(float) * kernel_size;
  size_t output_size_byte = sizeof(float) * output_size;

  float *img = (float*) malloc (image_size_byte);
  float *offsets_h = (float*) malloc (offset_size_byte);
  float *offsets_v = (float*) malloc (offset_size_byte);
  float *kernel = (float*) malloc (kernel_size_byte);
  float *output = (float*) malloc (output_size_byte);
  float *output_ref = (float*) malloc (output_size_byte);

  unsigned long long seed = 123;
  for (size_t i = 0; i < image_size; i++) img[i] = (unsigned char)(256*LCG_random_double(&seed));
  for (size_t i = 0; i < kernel_size; i++) kernel[i] = (unsigned char)(256*LCG_random_double(&seed));
  for (size_t i = 0; i < offset_size; i++) {
    offsets_h[i] = LCG_random_double(&seed);
    offsets_v[i] = LCG_random_double(&seed);
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_img = sycl::malloc_device<float>(image_size, q);
  q.memcpy(d_img, img, image_size_byte);

  float *d_offsets_h = sycl::malloc_device<float>(offset_size, q);
  q.memcpy(d_offsets_h, offsets_h, offset_size_byte);

  float *d_offsets_v = sycl::malloc_device<float>(offset_size, q);
  q.memcpy(d_offsets_v, offsets_v, offset_size_byte);

  float *d_kernel = sycl::malloc_device<float>(kernel_size, q);
  q.memcpy(d_kernel, kernel, kernel_size_byte);

  float *d_output = sycl::malloc_device<float>(output_size, q);

  sycl::range<1> gws ((output_size + 255) / 256 * 256);
  sycl::range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class downsampling>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        car(item,
            d_img,
            d_kernel,
            d_offsets_h,
            d_offsets_v,
            d_output,
            p,
            1, // offset_unit,
            padding,
            output_size);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", time * 1e-9f / repeat);

  q.memcpy(output, d_output, output_size_byte).wait();
  reference (img, kernel, offsets_h, offsets_v, output_ref, p, 1, padding);

  const float tolerance = 1e-1f;
  float rmse = 0;
  for (size_t i = 0; i < output_size; i++)
    rmse += (output_ref[i] - output[i]) * (output_ref[i] - output[i]);
  rmse = sqrtf(rmse / output_size);
  printf("RMSE: %f\n", rmse);
  printf("%s\n", (rmse <= tolerance) ? "PASS" : "FAIL");

  sycl::free(d_img, q);
  sycl::free(d_offsets_h, q);
  sycl::free(d_offsets_v, q);
  sycl::free(d_kernel, q);
  sycl::free(d_output, q);

  free(img);
  free(offsets_h);
  free(offsets_v);
  free(kernel);
  free(output);
  free(output_ref);
  return 0;
}

