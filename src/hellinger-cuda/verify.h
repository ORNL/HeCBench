#ifdef VERIFY
#include <limits>
#include <cstdlib>

/**
 * Perform matrix multiplication on host to verify results from device.
 */
bool ValueSame(FP a, FP b) {
  return FABS(a - b) < 1e-5;
}

void VerifyResult(FP (*a_host)[N], FP (*b_host)[P], 
                  FP (*c_host)[P], FP (*c_back)[P]) {
  // Check that the results are correct by comparing with host computing.
  int i, j, k;

  // c_host is initialized to zero.
  for (i = 0; i < M; i++)
    for (j = 0; j < P; j++) c_host[i][j] = (FP)0.0;

  for (i = 0; i < M; i++) {
    for (k = 0; k < N; k++) {
      for (j = 0; j < P; j++) {
        c_host[i][j] += SQRT(a_host[i][k] * b_host[k][j]);
      }
    }
  }
  for (i = 0; i < M; i++) {
    for (j = 0; j < P; j++) {
      FP value = ((FP)1.0 - c_host[i][j]);
      FP gate = (!std::signbit(value));  // std::signbit is a host function
      c_host[i][j] = SQRT(gate * value);
    }
  }

  bool mismatch_found = false;

  // Compare host side results with the result buffer from device side: print
  // mismatched data 5 times only.
  int print_count = 0;

  for (i = 0; i < M; i++) {
    for (j = 0; j < P; j++) {
      if (!ValueSame(c_back[i][j], c_host[i][j])) {
        std::cout << "Fail - The result is incorrect for element: [" << i << ", "
             << j << "], expected: " << c_host[i][j]
             << ", but found: " << c_back[i][j] << "\n";
        mismatch_found = true;
        print_count++;
        if (print_count == 5) break;
      }
    }

    if (print_count == 5) break;
  }

  if (!mismatch_found) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
    exit(1);
  }
}
#endif
