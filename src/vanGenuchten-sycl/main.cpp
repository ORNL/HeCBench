#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

void vanGenuchten(
  const double *__restrict__ Ksat,
  const double *__restrict__ psi,
        double *__restrict__ C,
        double *__restrict__ theta,
        double *__restrict__ K,
  const int size,
  sycl::nd_item<1> &item)
{
  double Se, _theta, _psi, lambda, m;

  int i = item.get_global_id(0);
  if (i < size)
  {
    lambda = n - 1.0;
    m = lambda/n;

    // Compute the volumetric moisture content [eqn 21]
    _psi = psi[i] * 100.0;
    if ( _psi < 0.0 )
      _theta = (theta_S - theta_R) / sycl::pow(
               1.0 + sycl::pow((alpha * (-_psi)), n), m) + theta_R;
    else
      _theta = theta_S;

    theta[i] = _theta;

    // Compute the effective saturation [eqn 2]
    Se = (_theta - theta_R)/(theta_S - theta_R);

    // Compute the hydraulic conductivity [eqn 8]
    double t = 1.0 - sycl::pow(1.0 - sycl::pow(Se, 1.0 / m), m);
    K[i] = Ksat[i] * sycl::sqrt(Se) * t * t;

    // Compute the specific moisture storage derivative of eqn (21).
    // So we have to calculate C = d(theta)/dh. Then the unit is converted into [1/m].
    if (_psi < 0.0)
      C[i] = 100.0 * alpha * n * (1.0 / n - 1.0) *
             sycl::pow(alpha * sycl::fabs(_psi), n - 1.0) *
             (theta_R - theta_S) *
             sycl::pow(sycl::pow(alpha * sycl::fabs(_psi), n) + 1.0,
                            1.0 / n - 2.0);
    else
      C[i] = 0.0;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: ./%s <dimX> <dimY> <dimZ> <repeat>\n", argv[0]);
    return 1;
  }

  const int dimX = atoi(argv[1]);
  const int dimY = atoi(argv[2]);
  const int dimZ = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int size = dimX * dimY * dimZ;
  const int size_byte = size * sizeof(double);

  double *Ksat, *psi, *C, *theta, *K;
  double *C_ref, *theta_ref, *K_ref;

  Ksat = new double[size];
  psi = new double[size];
  C = new double[size];
  theta = new double[size];
  K = new double[size];

  C_ref = new double[size];
  theta_ref = new double[size];
  K_ref = new double[size];

  // arbitrary numbers
  for (int i = 0; i < size; i++) {
    Ksat[i] = 1e-6 +  (1.0 - 1e-6) * i / size;
    psi[i] = -100.0 + 101.0 * i / size;
  }

  // for verification
  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  double *d_Ksat, *d_psi, *d_C, *d_theta, *d_K;
  d_Ksat = (double *)sycl::malloc_device(size_byte, q);
  d_psi = (double *)sycl::malloc_device(size_byte, q);
  d_C = (double *)sycl::malloc_device(size_byte, q);
  d_theta = (double *)sycl::malloc_device(size_byte, q);
  d_K = (double *)sycl::malloc_device(size_byte, q);

  q.memcpy(d_Ksat, Ksat, size_byte);
  q.memcpy(d_psi, psi, size_byte);

  sycl::range<1> gws ((size + 255) / 256 * 256);
  sycl::range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        vanGenuchten(d_Ksat, d_psi, d_C, d_theta, d_K, size, item);
      });
    });

  q.wait();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(C, d_C, size_byte);
  q.memcpy(theta, d_theta, size_byte);
  q.memcpy(K, d_K, size_byte);

  q.wait();

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (fabs(C[i] - C_ref[i]) > 1e-3 ||
        fabs(theta[i] - theta_ref[i]) > 1e-3 ||
        fabs(K[i] - K_ref[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_Ksat, q);
  sycl::free(d_psi, q);
  sycl::free(d_C, q);
  sycl::free(d_theta, q);
  sycl::free(d_K, q);

  delete[] Ksat;
  delete[] psi;
  delete[] C;
  delete[] theta;
  delete[] K;
  delete[] C_ref;
  delete[] theta_ref;
  delete[] K_ref;

  return 0;
}
