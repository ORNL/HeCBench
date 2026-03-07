#include <cmath>
#include <cstdlib>
#include <iostream>
#include "degrid.h"
#include "kernels.cpp"

void init_gcf(PRECISION2 *gcf, size_t size) {
  for (size_t sub_x=0; sub_x<GCF_GRID; sub_x++ )
    for (size_t sub_y=0; sub_y<GCF_GRID; sub_y++ )
      for(size_t x=0; x<size; x++)
        for(size_t y=0; y<size; y++) {
          PRECISION tmp = std::sin(6.28*x/size/GCF_GRID)*std::exp(-(1.0*x*x+1.0*y*y*sub_y)/size/size/2);
          gcf[size*size*(sub_x+sub_y*GCF_GRID)+x+y*size].x() = tmp*std::sin(1.0*x*sub_x/(y+1));
          gcf[size*size*(sub_x+sub_y*GCF_GRID)+x+y*size].y() = tmp*std::cos(1.0*x*sub_x/(y+1));
        }
}

void degridCPU(PRECISION2 *out, PRECISION2 *in, PRECISION2 *img, PRECISION2 *gcf) {
  //degrid on the CPU
  //  out (out) - the output values for each location
  //  in  (in)  - the locations to be interpolated 
  //  img (in) - the image
  //  gcf (in) - the gridding convolution function

  //offset gcf to point to the middle for cleaner code later
  gcf += GCF_DIM*(GCF_DIM+1)/2;
  for(size_t n=0; n<NPOINTS; n++) {
    int sub_x = std::floor(GCF_GRID*(in[n].x()-std::floor(in[n].x())));
    int sub_y = std::floor(GCF_GRID*(in[n].y()-std::floor(in[n].y())));
    int main_x = std::floor(in[n].x());
    int main_y = std::floor(in[n].y());
    PRECISION sum_r = 0.0;
    PRECISION sum_i = 0.0;
    for (int a=-GCF_DIM/2; a<GCF_DIM/2 ;a++)
      for (int b=-GCF_DIM/2; b<GCF_DIM/2 ;b++) {
        PRECISION r1 = img[main_x+a+IMG_SIZE*(main_y+b)].x(); 
        PRECISION i1 = img[main_x+a+IMG_SIZE*(main_y+b)].y(); 
        PRECISION r2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].x();
        PRECISION i2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].y();
        if (main_x+a >= 0 && main_y+b >= 0 && 
            main_x+a < IMG_SIZE && main_y+b < IMG_SIZE) {
          sum_r += r1*r2 - i1*i2; 
          sum_i += r1*i2 + r2*i1;
        }
      }
    out[n].x() = sum_r;
    out[n].y() = sum_i;
  } 
  gcf -= GCF_DIM*(GCF_DIM+1)/2;
}

template <class T,class Thalf>
int w_comp_sub(const void* A, const void* B) {
  Thalf quota, rema, quotb, remb;
  rema = modf((*((T*)A)).x(), &quota);
  remb = modf((*((T*)B)).x(), &quotb);
  int sub_xa = (int) (GCF_GRID*rema);
  int sub_xb = (int) (GCF_GRID*remb);
  rema = modf((*((T*)A)).y(), &quota);
  remb = modf((*((T*)B)).y(), &quotb);
  int suba = (int) (GCF_GRID*rema) + GCF_GRID*sub_xa;
  int subb = (int) (GCF_GRID*remb) + GCF_GRID*sub_xb;
  if (suba > subb) return 1;
  if (suba < subb) return -1;
  return 0;
}

int main(void) {

  long img_size = (IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM)*sizeof(PRECISION2);
  long io_size = sizeof(PRECISION2)*NPOINTS;

  PRECISION2* out = (PRECISION2*) malloc(io_size);
  PRECISION2* in = (PRECISION2*) malloc(io_size);
  PRECISION2 *img = (PRECISION2*) malloc(img_size);
  PRECISION2 *gcf = (PRECISION2*) malloc(64*GCF_DIM*GCF_DIM*sizeof(PRECISION2));

  std::cout << "img size in bytes: " << img_size << std::endl;
  std::cout << "out size in bytes: " << io_size << std::endl;

  //img is padded (above and below) to avoid overruns
  img += IMG_SIZE*GCF_DIM+GCF_DIM;

  init_gcf(gcf, GCF_DIM);
  srand(2541617);
  for(size_t n=0; n<NPOINTS; n++) {
    in[n].x() = ((float)rand())/(float)RAND_MAX*8000;
    in[n].y() = ((float)rand())/(float)RAND_MAX*8000;
  }
  for(size_t x=0; x<IMG_SIZE;x++)
    for(size_t y=0; y<IMG_SIZE;y++) {
      img[x+IMG_SIZE*y].x() = std::exp(-((x-1400.0)*(x-1400.0)+(y-3800.0)*(y-3800.0))/8000000.0)+1.0;
      img[x+IMG_SIZE*y].y() = 0.4;
    }
  //Zero the data in the offset areas
  for (int x=-IMG_SIZE*GCF_DIM-GCF_DIM;x<0;x++) {
    img[x].x() = 0.0; img[x].y() = 0.0;
  }
  for (int x=0;x<IMG_SIZE*GCF_DIM+GCF_DIM;x++) {
    img[x+IMG_SIZE*IMG_SIZE].x() = 0.0; img[x+IMG_SIZE*IMG_SIZE].y() = 0.0;
  }

  std::qsort(in, NPOINTS, sizeof(PRECISION2), w_comp_sub<PRECISION2,PRECISION>);

  std::cout << "Computing on GPU..." << std::endl;
  degridGPU(out,in,img,gcf);

  std::cout << "Computing on CPU..." << std::endl;
  PRECISION2 *out_cpu=(PRECISION2*)malloc(sizeof(PRECISION2)*NPOINTS);
  degridCPU(out_cpu,in,img,gcf);

  std::cout << "Checking results against CPU:" << std::endl;
  PRECISION EPS = (sizeof(PRECISION) == sizeof(double)) ? 1e-7 : 1e-1;
  std::cout << "Error bound: " << EPS << std::endl;

  bool ok = true;
  for (size_t n = 0; n < NPOINTS; n++) {
    if (std::fabs(out[n].x()-out_cpu[n].x()) > EPS ||
        std::fabs(out[n].y()-out_cpu[n].y()) > EPS ) {
      ok = false;
      std::cout << n << ": F(" << in[n].x() << ", " << in[n].y() << ") = " 
        << out[n].x() << ", " << out[n].y() 
        << " vs. " << out_cpu[n].x() << ", " << out_cpu[n].y() 
        << std::endl;
      break;
    }
  }

  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  img -= GCF_DIM + IMG_SIZE*GCF_DIM;

  free(out);
  free(in);
  free(img);
  free(gcf);

  if (!ok) return 1;
  return 0;
}
