#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

#define GPU_THREADS 256

#define KERNEL_LOOP(index, range) \
   for (int index = 0; index < (range); index++)

template <typename T>
void sequenceMaskKernel(
    int N,
    int M,
    int B,
    const T* in,
    const int* seq_lengths,
    T fill_val,
    T* out)
{
  if (B >= 0) {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);
      int ind = N * M * i + M * j + k;
      out[ind] = (k >= seq_lengths[j] ? fill_val : in[ind]);
    }
  } else {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N/GPU_THREADS) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;
      out[index] = (j >= seq_lengths[i] ? fill_val : in[index]);
    }
  }
}

template <typename T>
void windowMaskKernel(
    int N,
    int M,
    int B,
    const T* in,
    const int* window_centers,
    const int radius,
    T fill_val,
    T* out) {
  if (B >= 0) {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] =
          (k < window_centers[j] - radius || k > window_centers[j] + radius
               ? fill_val
               : in[ind]);
    }
  } else {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N/GPU_THREADS) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] =
          (j < window_centers[i] - radius || j > window_centers[i] + radius
               ? fill_val
               : in[index]);
    }
  }
}

template <typename T>
void
upperMaskKernel(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k > j ? fill_val : in[ind]);
    }
  } else {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N/GPU_THREADS) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j > i ? fill_val : in[index]);
    }
  }
}

template <typename T>
void
lowerMaskKernel(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k < j ? fill_val : in[ind]);
    }
  } else {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N/GPU_THREADS) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j < i ? fill_val : in[index]);
    }
  }
}

template <typename T>
void
upperDiagMaskKernel(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k >= j ? fill_val : in[ind]);
    }
  } else {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N/GPU_THREADS) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j >= i ? fill_val : in[index]);
    }
  }
}

template <typename T>
void
lowerDiagMaskKernel(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k <= j ? fill_val : in[ind]);
    }
  } else {
    #pragma omp target teams distribute parallel for \
    num_teams(M*N/GPU_THREADS) num_threads(GPU_THREADS)
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j <= i ? fill_val : in[index]);
    }
  }
}

template<typename T>
void print_mask_ratio (T *h_out, T *out_ref, T fill_val, int data_size) {
  #pragma omp target update from (h_out[0:data_size])
  int error = memcmp(h_out, out_ref, data_size * sizeof(T));
  int cnt_fill = 0;
  for (int i = 0; i < data_size; i++) {
    if (h_out[i] == fill_val) cnt_fill++;
  }
  printf("%s, Mask ratio: %f\n", (error ? "FAIL" : "PASS"),
                                 (float) cnt_fill / data_size);
}

template<typename T>
void eval_mask (const int M, const int N, const int B, const int repeat) {

  const T fill_val = -1;
  const int radius = M / 4;  // closely related to mask ratio

  int batch_dim = (B <= 0) ? 1 : B;

  printf("\nM = %d, N = %d, B = %d\n", M, N, batch_dim);

  int data_size = N * M * batch_dim;
  size_t data_size_in_bytes = data_size * sizeof(T); 

  int window_size = N;
  size_t window_size_in_bytes = N * sizeof(int);

  int seq_len = N;
  size_t seq_len_in_bytes = seq_len * sizeof(int); 

  T *h_in = (T*) malloc (data_size_in_bytes);
  T *h_out = (T*) malloc (data_size_in_bytes);
  T *out_ref = (T*) malloc (data_size_in_bytes);
  int *h_seq_len = (int*) malloc (seq_len_in_bytes);
  int *h_window = (int*) malloc (window_size_in_bytes);

  srand(123);
  for (int i = 0; i < seq_len; i++) {
    h_seq_len[i] = rand() % (M / 2); // closely related to mask ratio
  }
  for (int i = 0; i < window_size; i++) {
    h_window[i] = rand() % M;
  }
  for (int i = 0; i < data_size; i++) {
    h_in[i] = rand() % (M * N);
  }
  
  #pragma omp target data map (to: h_in[0:data_size], \
                                   h_seq_len[0:seq_len], \
                                   h_window[0:window_size]) \
                          map (alloc: h_out[0:data_size])
  {
    sequenceMaskKernel_cpu(N, M, batch_dim, h_in, h_seq_len, fill_val, out_ref);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      sequenceMaskKernel(
        N, M, batch_dim, h_in, h_seq_len, fill_val, h_out);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of sequenceMask kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    print_mask_ratio(h_out, out_ref, fill_val, data_size);

    windowMaskKernel_cpu(N, M, batch_dim, h_in, h_window, radius, fill_val, out_ref);
 
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      windowMaskKernel(
        N, M, batch_dim, h_in, h_window, radius, fill_val, h_out);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of windowMask kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    print_mask_ratio(h_out, out_ref, fill_val, data_size);

    upperMaskKernel_cpu(N, M, batch_dim, h_in, fill_val, out_ref);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      upperMaskKernel(
        N, M, batch_dim, h_in, fill_val, h_out);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of upperMask kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    print_mask_ratio(h_out, out_ref, fill_val, data_size);

    lowerMaskKernel_cpu(N, M, batch_dim, h_in, fill_val, out_ref);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      lowerMaskKernel(
        N, M, batch_dim, h_in, fill_val, h_out);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of lowerMask kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    print_mask_ratio(h_out, out_ref, fill_val, data_size);

    upperDiagMaskKernel_cpu(N, M, batch_dim, h_in, fill_val, out_ref);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      upperDiagMaskKernel(
        N, M, batch_dim, h_in, fill_val, h_out);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of upperDiagMask kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    print_mask_ratio(h_out, out_ref, fill_val, data_size);

    lowerDiagMaskKernel_cpu(N, M, batch_dim, h_in, fill_val, out_ref);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      lowerDiagMaskKernel(N, M, batch_dim, h_in, fill_val, h_out);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of lowerDiagMask kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    print_mask_ratio(h_out, out_ref, fill_val, data_size);
  }

  free(h_in);
  free(h_out);
  free(out_ref);
  free(h_window);
  free(h_seq_len);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <sequence length> <sequence length> <batch size> <repeat>\n", argv[0]);
    return 1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int B = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  eval_mask<int>(M, N, B, repeat);

  if ((error) return 1;
  return 0;
}
