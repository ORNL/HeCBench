#include <cstring>
#include <cstdlib>

void L2_norm_ref (const float *x, float &l2_norm, int n)
{
  l2_norm = 0;
  for (int i = 0; i < n; i++) {
    l2_norm += x[i]*x[i];
  }
}

void compute_ref (
    const float * __restrict__ x,
          float * __restrict__ grad,
    const int   * __restrict__ A_row_ptr,
    const int   * __restrict__ A_col_index,
    const float * __restrict__ A_value,
    const int   * __restrict__ A_y_label,
    float  &total_obj_val,
    int    &correct,
    int m )
{
  for (int i = 0; i < m; i++) {
    // Simple sparse matrix multiply x' = A * x
    float xp = 0.f;
    for( int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      xp += A_value[j] * x[A_col_index[j]];
    }

    // compute objective
    float v = logf(1.f + expf(-xp * A_y_label[i]));
    total_obj_val += v;

    // compute errors
    float prediction = 1.f / (1.f + expf(-xp));
    int t = (prediction >= 0.5f) ? 1 : -1;
    if (A_y_label[i] == t) correct++;

    // compute gradient at x
    float accum = expf(-A_y_label[i] * xp);
    accum = accum / (1.f + accum);
    for(int j = A_row_ptr[i]; j < A_row_ptr[i+1]; ++j){
      float temp = -accum * A_value[j] * A_y_label[i];
      grad[A_col_index[j]] += temp;
    }
  }
}

void update_ref(float * __restrict__ x,
                const float * __restrict__ grad,
                int m, int n, float lambda, float alpha)
{
  for (int i = 0; i < n; i++ ) {
    float g = grad[i] / (float)m + lambda * x[i];
    x[i] = x[i] - alpha * g;
  }
}

void reference (Classification_Data_CRS &A,
                std::vector<float> &x, std::vector<float> &grad,
                int m, int n, int iters, float alpha, float lambda,
                float obj_val, float train_error)
{
  float h_obj_val = 0.f;
  float h_train_error = 0.f;
  float *h_x;
  h_x = (float*) malloc (n * sizeof(float));
  memcpy(h_x, x.data(), n * sizeof(float));

  float *h_grad;
  h_grad = (float*) malloc (n * sizeof(float));

  int *h_row_ptr;
  h_row_ptr = (int*) malloc (A.row_ptr.size() * sizeof(int));
  memcpy(h_row_ptr, A.row_ptr.data(), A.row_ptr.size() * sizeof(int));

  int *h_col_index;
  h_col_index = (int*) malloc (A.col_index.size() * sizeof(int));
  memcpy(h_col_index, A.col_index.data(), A.col_index.size() * sizeof(int));

  float *h_value;
  h_value = (float*) malloc (A.values.size() * sizeof(float));
  memcpy(h_value, A.values.data(), A.values.size() * sizeof(float));

  int *h_y_label;
  h_y_label = (int*) malloc (A.y_label.size() * sizeof(int));
  memcpy(h_y_label, A.y_label.data(), A.y_label.size() * sizeof(int));

  for (int k = 0; k < iters; k++) {

    // reset the training status
    float total_obj_val = 0.f;
    float l2_norm = 0.f;
    int correct = 0;

    //reset gradient vector
    std::fill(grad.begin(), grad.end(), 0.f);

    memcpy(h_grad, grad.data(), n * sizeof(float));

    // compute the total objective, correct rate, and gradient
    compute_ref(
        h_x,
        h_grad,
        h_row_ptr,
        h_col_index,
        h_value,
        h_y_label,
        total_obj_val,
        correct,
        m
      );

    // display training status for verification
    L2_norm_ref(h_x, l2_norm, n);

    h_obj_val = total_obj_val / (float)m + 0.5f * lambda * l2_norm;
    h_train_error = 1.f-(correct/(float)m);

    // update x (gradient does not need to be updated)
    update_ref(h_x, h_grad, m, n, lambda, alpha);
  }

  bool ok = (fabsf(obj_val - h_obj_val) < 1e-3f) &&
            (fabsf(train_error - h_train_error) < 1e-3f);
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(h_row_ptr);
  free(h_col_index);
  free(h_value);
  free(h_y_label);
  free(h_x);
  free(h_grad);
}
