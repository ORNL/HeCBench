template <typename T>
struct Task;

// simple host dgemv: assume data is in row-major format and square
template <typename T>
void gemv(int m, int n, T alpha, T *A, T *x, T beta, T *result) {
  // rows
  for (int i = 0; i < m; i++) {
    result[i] *= beta;

    for (int j = 0; j < n; j++) {
      result[i] += A[i * n + j] * x[j];
    }
  }
}

// simple dgemv on a device
template <typename T>
void ref_gemv(int m, int n, T alpha, T *A, T *x, T beta, T *result) {
  // cols
  for (int i = 0; i < n; i++) {
    result[i] *= beta;

    for (int j = 0; j < m; j++) {
      result[i] += A[j * n + i] * x[j];
    }
  }
}

template <typename T>
void check(std::vector<Task<T> > &TaskList) {
  bool ok = true;
  for (size_t i = 0; i < TaskList.size(); i++) {
    auto tl = TaskList[i];
    if (tl.size >= 100) {
      T *ref_result = (T*) malloc (tl.size * sizeof(T));
      ref_gemv(tl.size, tl.size, (T)1.0, tl.data, tl.vector, (T)0.0, ref_result);
      for (size_t j = 0; j < tl.size; j++) {
        if (fabs(tl.result[j] - ref_result[j]) > 1e-3) {
          ok = false;
          break;
        }
      }
      free(ref_result);
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
}
