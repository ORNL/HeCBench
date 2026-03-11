float softplus_f_ref(float x) {
  return (x > 20.f) ? x : logf(1.f + expf(x));
}
float silu_f_ref(float x) {
  return x / (1.f + expf(-x));
}

void selective_scan_ref(
    const float* u,          // (B, D, L)
    const float* delta,      // (B, D, L)
    const float* A,          // (D, N)
    const float* B,          // (B, N, L)
    const float* C,          // (B, N, L)
    const float* D_vec,      // (D,)
    const float* delta_bias, // (D,)
    const float* z,          // (B, D, L) or nullptr
    bool   delta_plus,
    int    batch, int dim, int dstate, int seqlen,
    float* ssm_states,       // (B, D, N)  in/out
    float* y                 // (B, D, L)  out
)
{
  float *h = (float*) malloc (dstate * sizeof(float));
  for (int b = 0; b < batch; ++b) {
    for (int d = 0; d < dim; ++d) {
      for (int n = 0; n < dstate; ++n)
        h[n] = ssm_states[b*dim*dstate + d*dstate + n];

      float db = delta_bias[d];
      float Dv = D_vec[d];

      for (int l = 0; l < seqlen; ++l) {
        float u_v  = u    [b*dim*seqlen + d*seqlen + l];
        float dt   = delta[b*dim*seqlen + d*seqlen + l] + db;
        if (delta_plus) dt = softplus_f_ref(dt);

        float yv = Dv * u_v;
        for (int n = 0; n < dstate; ++n) {
          float dA = expf(dt * A[d*dstate + n]);
          float dB = dt  * B[b*dstate*seqlen + n*seqlen + l];
          h[n] = dA * h[n] + dB * u_v;
          yv  += C[b*dstate*seqlen + n*seqlen + l] * h[n];
        }
        if (z) yv *= silu_f_ref(z[b*dim*seqlen + d*seqlen + l]);
        y[b*dim*seqlen + d*seqlen + l] = yv;
      }
      for (int n = 0; n < dstate; ++n)
        ssm_states[b*dim*dstate + d*dstate + n] = h[n];
    }
  }
  free(h);
}
