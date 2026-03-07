#include "morphology.h"
#include <cstdlib>

void display(unsigned char *img, const int height, const int width)
{
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++)
      printf("%d ", img[i*width+j]);
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    printf("Usage: %s <kernel width> <kernel height> ", argv[0]);
    printf("<image width> <image height> <repeat>\n");
    return 1;
  }

  int hsize = atoi(argv[1]);  // kernel width
  int vsize = atoi(argv[2]);  // kernel height
  int width = atoi(argv[3]);  // image width
  int height = atoi(argv[4]); // image height
  int repeat = atoi(argv[5]);

  unsigned int memSize = width * height * sizeof(unsigned char);
  unsigned char* srcImg = (unsigned char*) malloc (memSize);

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      srcImg[i*width+j] = (i == (height/2 - 1) &&
                           j == (width/2 - 1)) ? WHITE : BLACK;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  unsigned char *img_d = sycl::malloc_device<unsigned char>(memSize, q);
  q.memcpy(img_d, srcImg, memSize);

  unsigned char *tmp_d = sycl::malloc_device<unsigned char>(memSize, q);

  double dilate_time = 0.0, erode_time = 0.0;

  for (int n = 0; n < repeat; n++) {
    dilate_time += dilate(q, img_d, tmp_d, width, height, hsize, vsize);
    erode_time += erode(q, img_d, tmp_d, width, height, hsize, vsize);
  }

  printf("Average kernel execution time (dilate): %f (s)\n", (dilate_time * 1e-9f) / repeat);
  printf("Average kernel execution time (erode): %f (s)\n", (erode_time * 1e-9f) / repeat);

  q.memcpy(srcImg, img_d, memSize).wait();

  int s = 0;
  for (unsigned int i = 0; i < memSize; i++) s += srcImg[i];
  printf("%s\n", s == WHITE ? "PASS" : "FAIL");

  sycl::free(img_d, q);
  sycl::free(tmp_d, q);
  free(srcImg);
  if (s != WHITE) return 1;
  return 0;
}
