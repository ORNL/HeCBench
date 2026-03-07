//! Thread-local Matrix-Vector multiplication.
template <int n>
void Mv_l_ref(const double* A, const double* v, double* out)
{
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = sum;
  }
}

template <int n>
void Mv_l_ref(double alpha, const double* A, const double* v, double* out)
{
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = alpha * sum;
  }
}

//! Thread-local Matrix-Matrix multiplication.
template <int n, bool aT = false, bool bT = false>
void MM_l_ref(const double* A, const double* B, double* out)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        double Aik = aT ? A[k + i * n] : A[i + k * n];
        double Bkj = bT ? B[j + k * n] : B[k + j * n];
        sum += Aik * Bkj;
      }
      out[i + j * n] = sum;
    }
  }
}

template <int rd>
void kalman_ref (
  const double*__restrict__ ys,
  int nobs,
  const double*__restrict__ T,
  const double*__restrict__ Z,
  const double*__restrict__ RQR,
  const double*__restrict__ P,
  const double*__restrict__ alpha,
  bool intercept,
  const double*__restrict__ d_mu,
  int batch_size,
  double*__restrict__ vs,
  double*__restrict__ Fs,
  double*__restrict__ sum_logFs,
  int n_diff,
  int fc_steps = 0,
  double*__restrict__ d_fc = nullptr,
  bool conf_int = false,
  double* d_F_fc = nullptr)
{
  for (int bid = 0; bid < batch_size; bid++) {
    constexpr int rd2 = rd * rd;
    double l_RQR[rd2] = {0.0};
    double l_T[rd2] = {0.0};
    double l_Z[rd] = {0.0};
    double l_P[rd2] = {0.0};
    double l_alpha[rd] = {0.0};
    double l_K[rd] = {0.0};
    double l_tmp[rd2] = {0.0};
    double l_TP[rd2] = {0.0};

    // Load global mem into registers
    int b_rd_offset  = bid * rd;
    int b_rd2_offset = bid * rd2;
    for (int i = 0; i < rd2; i++) {
      l_RQR[i] = RQR[b_rd2_offset + i];
      l_T[i]   = T[b_rd2_offset + i];
      l_P[i]   = P[b_rd2_offset + i];
    }
    for (int i = 0; i < rd; i++) {
      if (n_diff > 0) l_Z[i] = Z[b_rd_offset + i];
      l_alpha[i] = alpha[b_rd_offset + i];
    }

    double b_sum_logFs = 0.0;
    const double* b_ys = ys + bid * nobs;
    double* b_vs       = vs + bid * nobs;
    double* b_Fs       = Fs + bid * nobs;

    double mu = intercept ? d_mu[bid] : 0.0;

    for (int it = 0; it < nobs; it++) {
      // 1. v = y - Z*alpha
      double vs_it = b_ys[it];
      if (n_diff == 0)
        vs_it -= l_alpha[0];
      else {
        for (int i = 0; i < rd; i++) {
          vs_it -= l_alpha[i] * l_Z[i];
        }
      }
      b_vs[it] = vs_it;

      // 2. F = Z*P*Z'
      double _Fs;
      if (n_diff == 0)
        _Fs = l_P[0];
      else {
        _Fs = 0.0;
        for (int i = 0; i < rd; i++) {
          for (int j = 0; j < rd; j++) {
            _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
          }
        }
      }
      b_Fs[it] = _Fs;
      if (it >= n_diff) b_sum_logFs += log(_Fs);

      // 3. K = 1/Fs[it] * T*P*Z'
      // TP = T*P
      MM_l_ref<rd>(l_T, l_P, l_TP);
      // K = 1/Fs[it] * TP*Z'
      double _1_Fs = 1.0 / _Fs;
      if (n_diff == 0) {
        for (int i = 0; i < rd; i++) {
          l_K[i] = _1_Fs * l_TP[i];
        }
      } else
        Mv_l_ref<rd>(_1_Fs, l_TP, l_Z, l_K);

      // 4. alpha = T*alpha + K*vs[it] + c
      // tmp = T*alpha
      Mv_l_ref<rd>(l_T, l_alpha, l_tmp);
      // alpha = tmp + K*vs[it]
      for (int i = 0; i < rd; i++) {
        l_alpha[i] = l_tmp[i] + l_K[i] * vs_it;
      }
      // alpha = alpha + c
      l_alpha[n_diff] += mu;

      // 5. L = T - K * Z
      // L = T (L is tmp)
      for (int i = 0; i < rd2; i++) {
        l_tmp[i] = l_T[i];
      }
      // L = L - K * Z
      if (n_diff == 0) {
        for (int i = 0; i < rd; i++) {
          l_tmp[i] -= l_K[i];
        }
      } else {
        for (int i = 0; i < rd; i++) {
          for (int j = 0; j < rd; j++) {
            l_tmp[j * rd + i] -= l_K[i] * l_Z[j];
          }
        }
      }

      // 6. P = T*P*L' + R*Q*R'
      // P = TP*L'
      MM_l_ref<rd, false, true>(l_TP, l_tmp, l_P);
      // P = P + RQR
      for (int i = 0; i < rd2; i++) {
        l_P[i] += l_RQR[i];
      }
    }
    sum_logFs[bid] = b_sum_logFs;

    // Forecast
    double* b_fc   = fc_steps ? d_fc + bid * fc_steps : nullptr;
    double* b_F_fc = conf_int ? d_F_fc + bid * fc_steps : nullptr;
    for (int it = 0; it < fc_steps; it++) {
      if (n_diff == 0)
        b_fc[it] = l_alpha[0];
      else {
        double pred = 0.0;
        for (int i = 0; i < rd; i++) {
          pred += l_alpha[i] * l_Z[i];
        }
        b_fc[it] = pred;
      }

      // alpha = T*alpha + c
      Mv_l_ref<rd>(l_T, l_alpha, l_tmp);
      for (int i = 0; i < rd; i++) {
        l_alpha[i] = l_tmp[i];
      }
      l_alpha[n_diff] += mu;

      if (conf_int) {
        if (n_diff == 0)
          b_F_fc[it] = l_P[0];
        else {
          double _Fs = 0.0;
          for (int i = 0; i < rd; i++) {
            for (int j = 0; j < rd; j++) {
              _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
            }
          }
          b_F_fc[it] = _Fs;
        }

        // P = T*P*T' + RR'
        // TP = T*P
        MM_l_ref<rd>(l_T, l_P, l_TP);
        // P = TP*T'
        MM_l_ref<rd, false, true>(l_TP, l_T, l_P);
        // P = P + RR'
        for (int i = 0; i < rd2; i++) {
          l_P[i] += l_RQR[i];
        }
      }
    }
  }
}

template <int rd>
void reference (
  const int nseries,
  const int nobs,
  const double*__restrict__ ys,
  const double*__restrict__ T,
  const double*__restrict__ Z,
  const double*__restrict__ RQR,
  const double*__restrict__ P,
  const double*__restrict__ alpha,
  const double*__restrict__ mu,
  const double* d_F_fc,
  bool intercept,
  int n_diff,
  int fc_steps = 0,
  bool conf_int = false)
{
  const int repeat = 1;
  const int batch_size = nseries;
  const int nobs_size = nseries * nobs * sizeof(double);
  const int ns_size = nseries * sizeof(double);
  const int fc_size = fc_steps * nseries * sizeof(double);

  int i;

  double *vs = (double*) malloc (nobs_size);
  double *Fs = (double*) malloc (nobs_size);
  double *sum_logFs = (double*) malloc (ns_size);
  double *fc = (double*) malloc (fc_size);
  double *F_fc = (double*) malloc (fc_size);

  for (i = 0; i < repeat; i++)
    kalman_ref<rd>(
      ys,
      nobs,
      T,
      Z,
      RQR,
      P,
      alpha,
      true, // intercept,
      mu,
      batch_size,
      vs,
      Fs,
      sum_logFs,
      n_diff,
      fc_steps,
      fc,
      true, // forcast
      F_fc);

  bool ok = true;
  for (i = 0; i < fc_steps * nseries; i++) {
    if (fabs(F_fc[i] - d_F_fc[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(fc);
  free(F_fc);
  free(sum_logFs);
  free(Fs);
  free(vs);
}

