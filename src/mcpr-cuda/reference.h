void verify(const double *probs, const double *probs_ref, int alphas_size) {
  bool error = false;
  for (int i = 0; i < alphas_size; i++) {
    if (fabs(probs[i] - probs_ref[i]) > 1e-3) {
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);
}

void reference(
  const double* __restrict__ alphas,
  const double* __restrict__ rands,
        double* __restrict__ probs,
  int n, int K, int M)
{
  for (int i = 0; i < n; i++) {
    double maxval;
    int m, k;
    int maxind;
    double M_d = (double) M;
    double w[21]; // w[K]

    for(k = 0; k < K; ++k){   // initialize probs (though already done on CPU)
      probs[i*K + k] = 0.0;
    }

    // core computations
    for(m = 0; m < M; ++m){   // loop over Monte Carlo iterations
      for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
        w[k] = alphas[i*K + k] + rands[m*K + k];
      }

      // determine which category has max W
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        }
      }
      probs[i*K + maxind] += 1.0;
    }

    // compute final proportions
    for(k = 0; k < K; ++k) {
      probs[i*K + k] /= M_d;
    }
  }
}

void reference_unitStrides(
  const double* __restrict__ alphas,
  const double* __restrict__ rands,
        double* __restrict__ probs,
  int n, int K, int M)
{
  for (int i = 0; i < n; i++) {
    double maxval;
    int m, k;
    int maxind;
    double M_d = (double) M;
    double w[21]; // w[K]

    for(k = 0; k < K; ++k){  // initialize probs (though already done on CPU)
      probs[k*n + i] = 0.0;
    }

    // core computations
    for(m = 0; m < M; ++m){    // loop over Monte Carlo iterations
      for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
        // with +i we now have unit strides in inner loop
        w[k] = alphas[k*n + i] + rands[k*M + m];
      }

      // determine which category has max W
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        }
      }
      probs[maxind*n + i] += 1.0;
    }

    // compute final proportions
    for(k = 0; k < K; ++k) {
      // unit strides
      probs[k*n + i] /= M_d;
    }
  }
}
