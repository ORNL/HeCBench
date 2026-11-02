#ifndef KDA_HOST_UTILS_H
#define KDA_HOST_UTILS_H

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifndef KDA_VERIFY_T
#define KDA_VERIFY_T 16
#endif

#ifndef KDA_WARMUP
#define KDA_WARMUP 100
#endif

inline bool parse_positive_int(const char* text, const char* name, int* value)
{
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' ||
      parsed <= 0 || parsed > INT_MAX) {
    std::fprintf(stderr, "Error: %s must be a positive integer (got '%s')\n",
                 name, text);
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

inline void* checked_malloc(size_t bytes, const char* name)
{
  void* ptr = std::malloc(bytes);
  if (ptr == nullptr) {
    std::fprintf(stderr, "Error: failed to allocate %zu bytes for %s\n",
                 bytes, name);
    std::exit(EXIT_FAILURE);
  }
  return ptr;
}

inline double max_abs_error(const float* actual, const float* expected,
                            size_t count)
{
  double maximum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
      return 1.0e300;
    maximum = std::fmax(maximum,
                        std::fabs(static_cast<double>(actual[i]) -
                                  static_cast<double>(expected[i])));
  }
  return maximum;
}

// Compare the first T_check time steps of tensors stored as [B, T, inner].
inline double max_abs_error_time_prefix(const float* actual, const float* expected,
                                        int B, int T, int T_check, size_t inner)
{
  double maximum = 0.0;
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T_check; ++t) {
      const size_t off = ((size_t)b * (size_t)T + (size_t)t) * inner;
      maximum = std::fmax(maximum, max_abs_error(actual + off, expected + off, inner));
    }
  }
  return maximum;
}

#endif
