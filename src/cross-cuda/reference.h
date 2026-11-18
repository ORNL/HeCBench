#include <cmath>
#include <type_traits>

// Host cross product of contiguous 3-vectors. The arithmetic is done in
// double so the reference is independent of the device rounding.
template <typename T>
void cross_reference(int nrows, T* out, const T* x1, const T* x2)
{
  for (int i = 0; i < nrows; ++i) {
    const double a0 = static_cast<double>(x1[3 * i]);
    const double a1 = static_cast<double>(x1[3 * i + 1]);
    const double a2 = static_cast<double>(x1[3 * i + 2]);
    const double b0 = static_cast<double>(x2[3 * i]);
    const double b1 = static_cast<double>(x2[3 * i + 1]);
    const double b2 = static_cast<double>(x2[3 * i + 2]);
    out[3 * i]     = static_cast<T>(a1 * b2 - a2 * b1);
    out[3 * i + 1] = static_cast<T>(a2 * b0 - a0 * b2);
    out[3 * i + 2] = static_cast<T>(a0 * b1 - a1 * b0);
  }
}

template <typename T>
inline bool cross_matches_reference(const T* got, const T* ref, int n)
{
  const double tol = std::is_same<T, float>::value ? 1e-4 : 1e-10;
  for (int i = 0; i < n; i++) {
    if (std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i])) > tol)
      return false;
  }
  return true;
}
