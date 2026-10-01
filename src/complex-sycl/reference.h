#include <cmath>
#include <complex>

void ref_complex_float (sycl::nd_item<1> &item, char* checkSum, int n)
{
  int i = item.get_global_id(0);
  if (i >= n) return; 
  uint64_t seed = 1ULL;
  seed = fast_forward_LCG(seed, i);
  float r1 = LCG_random_double(&seed);
  float r2 = LCG_random_double(&seed); 
  float r3 = LCG_random_double(&seed); 
  float r4 = LCG_random_double(&seed); 

  auto z1 = std::complex<float>(r1, r2);
  auto z2 = std::complex<float>(r3, r4);

  char s = std::fabs(std::abs(z1 * z2) - std::abs(z1) * std::abs(z2)) < 1e-3f;

  s += std::fabs(std::abs(z1 + z2) * std::abs(z1 + z2) -
             ((z1 + z2) * (std::conj(z1) + std::conj(z2))).real()) < 1e-3f; 

  s += std::fabs(std::abs(z1 - z2) * std::abs(z1 - z2) -
             ((z1 - z2) * (std::conj(z1) - std::conj(z2))).real()) < 1e-3f;

  s += std::fabs((z1 * std::conj(z2) + z2 * std::conj(z1)).real() -
             2.0f * (z1.real() * z2.real() + z1.imag() * z2.imag())) < 1e-3f;

  s += std::fabs(std::abs(std::conj(z1) / z2) -
             std::abs(std::conj(z1) / std::conj(z2))) < 1e-3f;

  checkSum[i] = s;
}

void ref_complex_double (sycl::nd_item<1> &item, char* checkSum, int n)
{
  int i = item.get_global_id(0);
  if (i >= n) return; 
  uint64_t seed = 1ULL;
  seed = fast_forward_LCG(seed, i);
  double r1 = LCG_random_double(&seed);
  double r2 = LCG_random_double(&seed); 
  double r3 = LCG_random_double(&seed); 
  double r4 = LCG_random_double(&seed); 

  auto z1 = std::complex<double>(r1, r2);
  auto z2 = std::complex<double>(r3, r4);

  char s = std::fabs(std::abs(z1 * z2) - std::abs(z1) * std::abs(z2)) < 1e-3;

  s += std::fabs(std::abs(z1 + z2) * std::abs(z1 + z2) -
             ((z1 + z2) * (std::conj(z1) + std::conj(z2))).real()) < 1e-3; 

  s += std::fabs(std::abs(z1 - z2) * std::abs(z1 - z2) -
             ((z1 - z2) * (std::conj(z1) - std::conj(z2))).real()) < 1e-3;

  s += std::fabs((z1 * std::conj(z2) + z2 * std::conj(z1)).real() -
             2.0 * (z1.real() * z2.real() + z1.imag() * z2.imag())) < 1e-3;

  s += std::fabs(std::abs(std::conj(z1) / z2) -
             std::abs(std::conj(z1) / std::conj(z2))) < 1e-3;

  checkSum[i] = s;
}
