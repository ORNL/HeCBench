// HIP/MIOpen implementation of StormCast's EDM Heun sampling loop.
//
// The production StormCast denoiser is a Song U-Net. Porting its serialized
// PyTorch graph and trained weights is a separate model-export project, so this
// benchmark uses the same production convolution backend and an equivalent
// high-throughput residual denoiser:
//
//   x0(x, sigma) = Conv3x3(SiLU(Conv3x3(x))) / (1 + sigma * sigma)
//
// Both convolutions are MIOpen forward convolutions. The wrapper retains the
// production EDM schedule and Heun predictor/corrector flow used by StormCast.
// The sampled tensor is the forecast state, [1, 99, 512, 640]: one sample,
// the 99 HRRR variables, and the HRRR crop. The packaged sampler calls
// edm_stochastic_heun with S_churn = 0, so a step adds no extra noise. The
// Song U-Net around this state uses 128-channel features and a 299-channel
// diffusion input; those widths belong to the network, not to this latent.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include "../heun-cuda/reference.h"


static constexpr float kSigmaMin = 0.002f;
static constexpr float kSigmaMax = 800.f;
static constexpr float kRho = 7.f;

#define HIP_CHECK(cmd)                                                        \
  do {                                                                        \
    const hipError_t err_ = (cmd);                                            \
    if (err_ != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,            \
              hipGetErrorString(err_));                                       \
      return 1;                                                               \
    }                                                                         \
  } while (0)

#define MIOPEN_CHECK(cmd)                                                     \
  do {                                                                        \
    const miopenStatus_t status_ = (cmd);                                     \
    if (status_ != miopenStatusSuccess) {                                     \
      fprintf(stderr, "MIOpen error %s:%d: %d\n", __FILE__, __LINE__,         \
              static_cast<int>(status_));                                     \
      return 1;                                                               \
    }                                                                         \
  } while (0)

struct Step {
  float t_cur;
  float t_next;
  float h;
  int correct;
};

static void edm_timesteps(float* t, int steps) {
  const float smax = powf(kSigmaMax, 1.f / kRho);
  const float smin = powf(kSigmaMin, 1.f / kRho);
  for (int i = 0; i < steps; ++i) {
    const float f = static_cast<float>(i) / static_cast<float>(steps - 1);
    t[i] = powf(smax + f * (smin - smax), kRho);
  }
  t[steps] = 0.f;
}

__global__ void silu_scale(float* x, size_t n, float sigma) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = x[i];
    x[i] = (v / (1.f + expf(-v))) / (1.f + sigma * sigma);
  }
}

// EDM's ODE RHS for an x0 predictor: (x - x0) / sigma.
__global__ void heun_predict(const float* x, const float* x0, float* x_prime,
                             size_t n, Step s) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float d_cur = (x[i] - x0[i]) / s.t_cur;
    x_prime[i] = x[i] + s.h * d_cur;
  }
}

__global__ void heun_correct(const float* x, const float* x0_cur,
                             const float* x0_prime, float* x_next, size_t n,
                             Step s) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float d_cur = (x[i] - x0_cur[i]) / s.t_cur;
    if (!s.correct) {
      x_next[i] = x[i] + s.h * d_cur;
    } else {
      const float d_prime = (x[i] + s.h * d_cur - x0_prime[i]) / s.t_next;
      x_next[i] = x[i] + 0.5f * s.h * (d_cur + d_prime);
    }
  }
}

class ConvDenoiser {
 public:
  ConvDenoiser(size_t elements, int batch, int channels, int height, int width)
      : elements_(elements),
        batch_(batch),
        channels_(channels),
        height_(height),
        width_(width) {}

  int initialize(const float* host_w1, const float* host_w2) {
    MIOPEN_CHECK(miopenCreate(&handle_));
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&x_desc_));
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&w_desc_));
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&y_desc_));
    MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc_));
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(x_desc_, miopenFloat, batch_,
                                              channels_, height_, width_));
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(w_desc_, miopenFloat, channels_,
                                              channels_, 3, 3));
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(y_desc_, miopenFloat, batch_,
                                              channels_, height_, width_));
    MIOPEN_CHECK(miopenInitConvolutionDescriptor(
        conv_desc_, miopenConvolution, 1, 1, 1, 1, 1, 1));

    const size_t activation_bytes = elements_ * sizeof(float);
    const size_t weight_bytes = static_cast<size_t>(channels_) * channels_ *
                                3 * 3 * sizeof(float);
    HIP_CHECK(hipMalloc(&hidden_, activation_bytes));
    HIP_CHECK(hipMalloc(&x0_, activation_bytes));
    HIP_CHECK(hipMalloc(&w1_, weight_bytes));
    HIP_CHECK(hipMalloc(&w2_, weight_bytes));
    HIP_CHECK(hipMemcpy(w1_, host_w1, weight_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(w2_, host_w2, weight_bytes, hipMemcpyHostToDevice));

    MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
        handle_, w_desc_, x_desc_, conv_desc_, y_desc_, &workspace_bytes_));
    if (workspace_bytes_ != 0) HIP_CHECK(hipMalloc(&workspace_, workspace_bytes_));

    // The search compiles and measures candidate kernels once. It is excluded
    // from benchmark timing, just as PyTorch/MIOpen's warm-up is.
    miopenConvAlgoPerf_t perf{};
    int count = 0;
    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        handle_, x_desc_, hidden_, w_desc_, w1_, conv_desc_, y_desc_, x0_, 1,
        &count, &perf, workspace_, workspace_bytes_, true));
    if (count != 1) {
      fprintf(stderr, "MIOpen returned no forward convolution solution\n");
      return 1;
    }
    algo_ = perf.fwd_algo;
    printf("MIOpen forward solution time %.3f ms, workspace %zu MiB\n",
           perf.time, workspace_bytes_ / (1024 * 1024));
    return 0;
  }

  int operator()(const float* x, float sigma) {
    constexpr float alpha = 1.f;
    constexpr float beta = 0.f;
    MIOPEN_CHECK(miopenConvolutionForward(handle_, &alpha, x_desc_, x, w_desc_,
                                           w1_, conv_desc_, algo_, &beta, y_desc_,
                                           hidden_, workspace_, workspace_bytes_));
    constexpr int threads = 256;
    const int blocks =
        static_cast<int>((elements_ + threads - 1) / static_cast<size_t>(threads));
    silu_scale<<<blocks, threads>>>(hidden_, elements_, sigma);
    HIP_CHECK(hipGetLastError());
    MIOPEN_CHECK(miopenConvolutionForward(handle_, &alpha, x_desc_, hidden_,
                                           w_desc_, w2_, conv_desc_, algo_, &beta,
                                           y_desc_, x0_, workspace_, workspace_bytes_));
    return 0;
  }

  float* output() const { return x0_; }

  ~ConvDenoiser() {
    if (workspace_) (void)hipFree(workspace_);
    if (w2_) (void)hipFree(w2_);
    if (w1_) (void)hipFree(w1_);
    if (x0_) (void)hipFree(x0_);
    if (hidden_) (void)hipFree(hidden_);
    if (conv_desc_) miopenDestroyConvolutionDescriptor(conv_desc_);
    if (y_desc_) miopenDestroyTensorDescriptor(y_desc_);
    if (w_desc_) miopenDestroyTensorDescriptor(w_desc_);
    if (x_desc_) miopenDestroyTensorDescriptor(x_desc_);
    if (handle_) miopenDestroy(handle_);
  }

 private:
  size_t elements_;
  int batch_;
  int channels_;
  int height_;
  int width_;
  miopenHandle_t handle_ = nullptr;
  miopenTensorDescriptor_t x_desc_ = nullptr;
  miopenTensorDescriptor_t w_desc_ = nullptr;
  miopenTensorDescriptor_t y_desc_ = nullptr;
  miopenConvolutionDescriptor_t conv_desc_ = nullptr;
  miopenConvFwdAlgorithm_t algo_{};
  size_t workspace_bytes_ = 0;
  void* workspace_ = nullptr;
  float *hidden_ = nullptr, *x0_ = nullptr, *w1_ = nullptr, *w2_ = nullptr;
};

int main(int argc, char** argv) {
  if (argc != 7) {
    fprintf(stderr,
            "Usage: %s <repeat> <batch> <channels> <height> <width> <steps>\n",
            argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int batch = atoi(argv[2]);
  const int channels = atoi(argv[3]);
  const int height = atoi(argv[4]);
  const int width = atoi(argv[5]);
  const int steps = atoi(argv[6]);
  // The EDM schedule interpolates between sigma_max and sigma_min, so it
  // needs at least two timesteps.
  if (repeat <= 0 || batch <= 0 || channels <= 0 || height <= 0 || width <= 0 ||
      steps < 2) {
    fprintf(stderr,
            "Usage: %s <repeat> <batch> <channels> <height> <width> <steps>\n"
            "All arguments must be positive and steps must be at least 2\n",
            argv[0]);
    return 1;
  }

  const size_t elements =
      static_cast<size_t>(batch) * channels * height * width;
  const size_t bytes = elements * sizeof(float);
  printf("EDM Heun with MIOpen residual denoiser: [%d, %d, %d, %d], %d steps\n",
         batch, channels, height, width, steps);

  std::vector<float> host_x(elements);
  for (size_t i = 0; i < elements; ++i)
    host_x[i] = kSigmaMax * (static_cast<float>(i % 8192) / 4096.f - 1.f);

  float *d_a = nullptr, *d_b = nullptr, *d_prime = nullptr, *d_x0_cur = nullptr;
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));
  HIP_CHECK(hipMalloc(&d_prime, bytes));
  HIP_CHECK(hipMalloc(&d_x0_cur, bytes));

  const size_t weight_n =
      static_cast<size_t>(channels) * channels * 3 * 3;
  std::vector<float> w1(weight_n), w2(weight_n), reference(elements);
  edm_fill_weights(w1.data(), weight_n, 1);
  edm_fill_weights(w2.data(), weight_n, 2);

  ConvDenoiser denoiser(elements, batch, channels, height, width);
  if (denoiser.initialize(w1.data(), w2.data()) != 0) return 1;

  std::vector<float> times(static_cast<size_t>(steps) + 1);
  edm_timesteps(times.data(), steps);

  printf("Computing host reference...\n");
  fflush(stdout);
  const auto reference_start = std::chrono::steady_clock::now();
  edm_heun_reference(host_x.data(), w1.data(), w2.data(), reference.data(),
                     times.data(), steps, batch, channels, height, width);
  const double reference_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    reference_start).count();
  printf("Host reference time: %.3f s\n", reference_s);

  constexpr int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);

  HIP_CHECK(hipMemcpy(d_a, host_x.data(), bytes, hipMemcpyHostToDevice));
  constexpr int kDenoiserRepeats = 10;
  HIP_CHECK(hipDeviceSynchronize());
  const auto denoiser_start = std::chrono::steady_clock::now();
  for (int r = 0; r < kDenoiserRepeats; ++r)
    if (denoiser(d_a, 1.f) != 0) return 1;
  HIP_CHECK(hipDeviceSynchronize());
  const double denoiser_ms =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
      denoiser_start).count() * 1000.0 / kDenoiserRepeats;
  const double conv_flops = 2.0 * static_cast<double>(batch) * channels *
                            channels * height * width * 9.0;
  printf("MIOpen denoiser time: %.6f ms  estimated Conv3x3 time: %.6f ms  "
         "%.2f TFLOP/s\n",
         denoiser_ms, denoiser_ms / 2.0,
         conv_flops / ((denoiser_ms / 2.0) * 1e9));

  auto sample = [&]() -> int {
    float* current = d_a;
    float* next = d_b;
    for (int i = 0; i < steps; ++i) {
      const Step s{times[i], times[i + 1], times[i + 1] - times[i],
                   times[i + 1] != 0.f};
      if (denoiser(current, s.t_cur) != 0) return 1;
      float* x0_cur = denoiser.output();
      heun_predict<<<blocks, threads>>>(current, x0_cur, d_prime, elements, s);
      HIP_CHECK(hipGetLastError());
      if (s.correct) {
        // The second denoiser call reuses its output allocation.
        HIP_CHECK(hipMemcpyAsync(d_x0_cur, x0_cur, bytes, hipMemcpyDeviceToDevice));
        if (denoiser(d_prime, s.t_next) != 0) return 1;
        heun_correct<<<blocks, threads>>>(current, d_x0_cur, denoiser.output(), next,
                                          elements, s);
      } else {
        heun_correct<<<blocks, threads>>>(current, x0_cur, x0_cur, next, elements,
                                          s);
      }
      HIP_CHECK(hipGetLastError());
      std::swap(current, next);
    }
    return 0;
  };

  // Warm-up
  HIP_CHECK(hipMemcpy(d_a, host_x.data(), bytes, hipMemcpyHostToDevice));
  for (int r = 0; r < 100; ++r) if (sample() != 0) return 1;
  HIP_CHECK(hipDeviceSynchronize());

  double seconds = 0;
  for (int r = 0; r < repeat; ++r) {
    HIP_CHECK(hipMemcpy(d_a, host_x.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    if (sample() != 0) return 1;
    HIP_CHECK(hipDeviceSynchronize());
    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              start).count();
  }

  // sample() writes d_b, d_a, ... and swaps once per step, so an even step
  // count leaves the latent in d_a and an odd count leaves it in d_b.
  const float* gpu_final = (steps % 2 == 0) ? d_a : d_b;
  std::vector<float> output(elements);
  HIP_CHECK(hipMemcpy(output.data(), gpu_final, output.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  bool finite = true;
  double sum = 0.0;
  double sum_sq = 0.0;
  for (float x : output) {
    finite &= std::isfinite(x);
    sum += x;
    sum_sq += static_cast<double>(x) * x;
  }
  printf("Average sampling time: %.6f s (%d denoiser calls: 2x MIOpen 3x3 conv)\n",
         seconds / repeat, 2 * steps - 1);
  // Direct convolution and the selected MIOpen algorithm accumulate in
  // different orders. The element bound absorbs that rounding. The relative
  // RMS bound still fails a trajectory that merely lands at the same scale.
  const double n = static_cast<double>(elements);
  double ref_sq = 0.0;
  for (float ref : reference) ref_sq += static_cast<double>(ref) * ref;
  const double ref_rms = std::sqrt(ref_sq / n);
  const double rtol = 1e-3;
  const double atol = rtol * ref_rms;
  double max_abs = 0.0;
  double err_sq = 0.0;
  bool pass = finite;
  for (size_t i = 0; i < elements; ++i) {
    const double ref = static_cast<double>(reference[i]);
    const double diff = static_cast<double>(output[i]) - ref;
    const double abs_diff = std::abs(diff);
    const double tol = atol + rtol * std::abs(ref);
    max_abs = std::max(max_abs, abs_diff);
    err_sq += diff * diff;
    if (!(abs_diff <= tol) || !std::isfinite(output[i])) pass = false;
  }
  const double rms = std::sqrt(err_sq / n);
  const double rel_rms = rms / std::max(ref_rms, 1e-12);
  if (rel_rms > 1e-2) pass = false;
  printf("output sum %.12e  l2 %.12e\n", sum, std::sqrt(sum_sq));
  printf("max abs error %.3e  rms %.3e  rel rms %.3e\n", max_abs, rms, rel_rms);
  printf("%s\n", pass ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_x0_cur));
  HIP_CHECK(hipFree(d_prime));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_a));
  return pass ? 0 : 1;
}
