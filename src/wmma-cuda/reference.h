// Element-wise comparison
template <typename T>
void compareEqual(T const *a, T const *b, uint32_t size,
                  double tolerance = 10.0) {
  double max_relative_error = 0.0;

  for (uint32_t i = 0; i < size; i++) {
    auto valA = a[i];
    auto valB = b[i];
    auto relative_error = fabs(valA - valB) / (fabs(valA) + fabs(valB) + 1.0);

    if (relative_error > max_relative_error ||
        relative_error != relative_error) {
      max_relative_error = relative_error;
    }
  }
  auto eps = std::numeric_limits<T>::epsilon();
  if (max_relative_error != max_relative_error ||
      max_relative_error > eps * tolerance) {
    std::cout << "FAILED\n";
    exit(1);
  } else {
    std::cout << "PASSED\n";
  }

  std::cout << "Max relative error: " << max_relative_error << std::endl;
}

// Host GEMM validation
void gemm_cpu_h(uint32_t m, uint32_t n, uint32_t k, fp16 const *a,
                fp16 const *b, fp32 const *c, fp32 *d,
                uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t ldd,
                fp32 alpha, fp32 beta) {
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      fp32 accum = 0.0f;
      for (uint32_t h = 0; h < k; ++h) {
        accum += static_cast<fp32>(a[i * lda + h]) *
                 static_cast<fp32>(b[j * ldb + h]);
      }
      d[i * ldd + j] = alpha * accum + beta * c[i * ldc + j];
    }
  }
}
