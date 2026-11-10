// SYCL/oneDNN implementation of StormCast's EDM Heun sampling loop.
//
// The production StormCast denoiser is a Song U-Net. Porting its serialized
// PyTorch graph and trained weights is a separate model-export project, so this
// benchmark uses the same production convolution backend and an equivalent
// high-throughput residual denoiser:
//
//   x0(x, sigma) = Conv3x3(SiLU(Conv3x3(x))) / (1 + sigma * sigma)
//
// Both convolutions are oneDNN forward convolutions. The wrapper retains the
// production EDM schedule and Heun predictor/corrector flow used by StormCast.
// The sampled tensor is the forecast state, [1, 99, 512, 640]: one sample,
// the 99 HRRR variables, and the HRRR crop. The packaged sampler calls
// edm_stochastic_heun with S_churn = 0, so a step adds no extra noise. The
// Song U-Net around this state uses 128-channel features and a 299-channel
// diffusion input; those widths belong to the network, not to this latent.
//
// The shared CUDA reference runs the same trajectory with a direct convolution so the
// device latent can be checked against the host.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include "../heun-cuda/reference.h"

static constexpr float kSigmaMin = 0.002f;
static constexpr float kSigmaMax = 800.f;
static constexpr float kRho = 7.f;

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

static void silu_scale(float* x, size_t n, float sigma, sycl::nd_item<1> item) {
  const size_t i = item.get_global_id(0);
  if (i < n) {
    const float v = x[i];
    x[i] = (v / (1.f + sycl::exp(-v))) / (1.f + sigma * sigma);
  }
}

// EDM's ODE RHS for an x0 predictor: (x - x0) / sigma.
static void heun_predict(const float* x, const float* x0, float* x_prime,
                         size_t n, Step s, sycl::nd_item<1> item) {
  const size_t i = item.get_global_id(0);
  if (i < n) {
    const float d_cur = (x[i] - x0[i]) / s.t_cur;
    x_prime[i] = x[i] + s.h * d_cur;
  }
}

static void heun_correct(const float* x, const float* x0_cur,
                         const float* x0_prime, float* x_next, size_t n, Step s,
                         sycl::nd_item<1> item) {
  const size_t i = item.get_global_id(0);
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
  ConvDenoiser(sycl::queue q, size_t elements, int batch, int channels,
               int height, int width)
      : q_(std::move(q)),
        elements_(elements),
        batch_(batch),
        channels_(channels),
        height_(height),
        width_(width) {}

  int initialize(const float* host_w1, const float* host_w2) {
    try {
      engine_ = dnnl::sycl_interop::make_engine(q_.get_device(), q_.get_context());
      stream_ = dnnl::sycl_interop::make_stream(engine_, q_);

      const dnnl::memory::dim n = batch_;
      const dnnl::memory::dim c = channels_;
      const dnnl::memory::dim h = height_;
      const dnnl::memory::dim w = width_;
      const dnnl::memory::dims act_dims = {n, c, h, w};
      const dnnl::memory::dims weight_dims = {c, c, 3, 3};
      const dnnl::memory::dims strides = {1, 1};
      const dnnl::memory::dims padding = {1, 1};
      user_src_md_ = dnnl::memory::desc(act_dims, dnnl::memory::data_type::f32,
                                        dnnl::memory::format_tag::nchw);
      user_dst_md_ = user_src_md_;
      user_w_md_ = dnnl::memory::desc(weight_dims, dnnl::memory::data_type::f32,
                                      dnnl::memory::format_tag::oihw);

      // Let oneDNN pick the convolution layout. Activations are reordered back
      // to NCHW so the SiLU and Heun kernels share the host reference layout.
      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      const auto conv_pd = dnnl::convolution_forward::primitive_desc(
          engine_, dnnl::prop_kind::forward_inference,
          dnnl::algorithm::convolution_direct,
          dnnl::memory::desc(act_dims, dnnl::memory::data_type::f32,
                             dnnl::memory::format_tag::any),
          dnnl::memory::desc(weight_dims, dnnl::memory::data_type::f32,
                             dnnl::memory::format_tag::any),
          dnnl::memory::desc(act_dims, dnnl::memory::data_type::f32,
                             dnnl::memory::format_tag::any),
          strides, padding, padding, attr);
      conv_ = dnnl::convolution_forward(conv_pd);

      const size_t weight_n =
          static_cast<size_t>(channels_) * channels_ * 3 * 3;
      const size_t weight_bytes = weight_n * sizeof(float);
      hidden_ = sycl::malloc_device<float>(elements_, q_);
      x0_ = sycl::malloc_device<float>(elements_, q_);
      w1_ = sycl::malloc_device<float>(weight_n, q_);
      w2_ = sycl::malloc_device<float>(weight_n, q_);
      if (!hidden_ || !x0_ || !w1_ || !w2_) {
        fprintf(stderr, "SYCL allocation failed\n");
        return 1;
      }
      q_.memcpy(w1_, host_w1, weight_bytes);
      q_.memcpy(w2_, host_w2, weight_bytes);
      q_.wait();

      reorder_src_ = conv_pd.src_desc() != user_src_md_;
      reorder_dst_ = conv_pd.dst_desc() != user_dst_md_;
      reorder_weights_ = conv_pd.weights_desc() != user_w_md_;
      if (reorder_src_) {
        conv_src_ = dnnl::memory(conv_pd.src_desc(), engine_);
        reorder_src_prim_ = dnnl::reorder(dnnl::reorder::primitive_desc(
            engine_, user_src_md_, engine_, conv_pd.src_desc()));
      }
      if (reorder_dst_) {
        conv_dst_ = dnnl::memory(conv_pd.dst_desc(), engine_);
        reorder_dst_prim_ = dnnl::reorder(dnnl::reorder::primitive_desc(
            engine_, conv_pd.dst_desc(), engine_, user_dst_md_));
      }
      w1_mem_ = pack_weights(w1_, conv_pd.weights_desc());
      w2_mem_ = pack_weights(w2_, conv_pd.weights_desc());

      const auto scratch_md = conv_pd.scratchpad_desc();
      use_scratchpad_ = scratch_md.get_size() > 0;
      if (use_scratchpad_) scratchpad_ = dnnl::memory(scratch_md, engine_);
      printf("oneDNN forward convolution: workspace %zu MiB%s\n",
             scratch_md.get_size() / (1024 * 1024),
             (reorder_src_ || reorder_dst_ || reorder_weights_)
                 ? ", with layout reorder"
                 : "");
    } catch (const dnnl::error& e) {
      fprintf(stderr, "oneDNN error: %s\n", e.what());
      return 1;
    } catch (const sycl::exception& e) {
      fprintf(stderr, "SYCL error: %s\n", e.what());
      return 1;
    }
    return 0;
  }

  int operator()(const float* x, float sigma) {
    try {
      convolve(x, w1_mem_, hidden_);
      const size_t threads = 256;
      const size_t blocks = (elements_ + threads - 1) / threads;
      float* hidden = hidden_;
      const size_t n = elements_;
      q_.parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                      [=](sycl::nd_item<1> item) {
                        silu_scale(hidden, n, sigma, item);
                      });
      convolve(hidden_, w2_mem_, x0_);
    } catch (const dnnl::error& e) {
      fprintf(stderr, "oneDNN error: %s\n", e.what());
      return 1;
    } catch (const sycl::exception& e) {
      fprintf(stderr, "SYCL error: %s\n", e.what());
      return 1;
    }
    return 0;
  }

  float* output() const { return x0_; }

  ~ConvDenoiser() {
    w1_mem_ = dnnl::memory();
    w2_mem_ = dnnl::memory();
    conv_src_ = dnnl::memory();
    conv_dst_ = dnnl::memory();
    scratchpad_ = dnnl::memory();
    conv_ = dnnl::convolution_forward();
    reorder_src_prim_ = dnnl::reorder();
    reorder_dst_prim_ = dnnl::reorder();
    stream_ = dnnl::stream();
    if (w2_) sycl::free(w2_, q_);
    if (w1_) sycl::free(w1_, q_);
    if (x0_) sycl::free(x0_, q_);
    if (hidden_) sycl::free(hidden_, q_);
  }

 private:
  dnnl::memory pack_weights(float* device_w, const dnnl::memory::desc& packed_md) {
    auto user = dnnl::sycl_interop::make_memory(
        user_w_md_, engine_, dnnl::sycl_interop::memory_kind::usm, device_w);
    if (!reorder_weights_) return user;
    dnnl::memory packed(packed_md, engine_);
    dnnl::reorder(dnnl::reorder::primitive_desc(engine_, user_w_md_, engine_,
                                                packed_md))
        .execute(stream_, {{DNNL_ARG_FROM, user}, {DNNL_ARG_TO, packed}});
    stream_.wait();
    return packed;
  }

  void convolve(const float* src, const dnnl::memory& weights, float* dst) {
    auto src_mem = dnnl::sycl_interop::make_memory(
        user_src_md_, engine_, dnnl::sycl_interop::memory_kind::usm,
        const_cast<float*>(src));
    auto dst_mem = dnnl::sycl_interop::make_memory(
        user_dst_md_, engine_, dnnl::sycl_interop::memory_kind::usm, dst);

    std::unordered_map<int, dnnl::memory> args;
    if (reorder_src_) {
      reorder_src_prim_.execute(
          stream_, {{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, conv_src_}});
      args.emplace(DNNL_ARG_SRC, conv_src_);
    } else {
      args.emplace(DNNL_ARG_SRC, src_mem);
    }
    args.emplace(DNNL_ARG_WEIGHTS, weights);
    if (reorder_dst_) {
      args.emplace(DNNL_ARG_DST, conv_dst_);
    } else {
      args.emplace(DNNL_ARG_DST, dst_mem);
    }
    if (use_scratchpad_) args.emplace(DNNL_ARG_SCRATCHPAD, scratchpad_);
    conv_.execute(stream_, args);
    if (reorder_dst_) {
      reorder_dst_prim_.execute(
          stream_, {{DNNL_ARG_FROM, conv_dst_}, {DNNL_ARG_TO, dst_mem}});
    }
  }

  sycl::queue q_;
  size_t elements_;
  int batch_;
  int channels_;
  int height_;
  int width_;
  dnnl::engine engine_;
  dnnl::stream stream_;
  dnnl::convolution_forward conv_;
  dnnl::reorder reorder_src_prim_;
  dnnl::reorder reorder_dst_prim_;
  dnnl::memory::desc user_src_md_;
  dnnl::memory::desc user_dst_md_;
  dnnl::memory::desc user_w_md_;
  dnnl::memory w1_mem_;
  dnnl::memory w2_mem_;
  dnnl::memory conv_src_;
  dnnl::memory conv_dst_;
  dnnl::memory scratchpad_;
  bool reorder_src_ = false;
  bool reorder_dst_ = false;
  bool reorder_weights_ = false;
  bool use_scratchpad_ = false;
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
  printf("EDM Heun with oneDNN residual denoiser: [%d, %d, %d, %d], %d steps\n",
         batch, channels, height, width, steps);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  std::vector<float> host_x(elements);
  for (size_t i = 0; i < elements; ++i)
    host_x[i] = kSigmaMax * (static_cast<float>(i % 8192) / 4096.f - 1.f);

  float* d_a = sycl::malloc_device<float>(elements, q);
  float* d_b = sycl::malloc_device<float>(elements, q);
  float* d_prime = sycl::malloc_device<float>(elements, q);
  float* d_x0_cur = sycl::malloc_device<float>(elements, q);
  if (!d_a || !d_b || !d_prime || !d_x0_cur) {
    fprintf(stderr, "SYCL allocation failed\n");
    return 1;
  }

  const size_t weight_n =
      static_cast<size_t>(channels) * channels * 3 * 3;
  std::vector<float> w1(weight_n), w2(weight_n), reference(elements);
  edm_fill_weights(w1.data(), weight_n, 1);
  edm_fill_weights(w2.data(), weight_n, 2);

  ConvDenoiser denoiser(q, elements, batch, channels, height, width);
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

  const size_t threads = 256;
  const size_t blocks = (elements + threads - 1) / threads;

  q.memcpy(d_a, host_x.data(), bytes).wait();
  constexpr int kDenoiserRepeats = 10;
  const auto denoiser_start = std::chrono::steady_clock::now();
  for (int r = 0; r < kDenoiserRepeats; ++r)
    if (denoiser(d_a, 1.f) != 0) return 1;
  q.wait();
  const double denoiser_ms =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
      denoiser_start).count() * 1000.0 / kDenoiserRepeats;
  const double conv_flops = 2.0 * static_cast<double>(batch) * channels *
                            channels * height * width * 9.0;
  printf("oneDNN denoiser time: %.6f ms  estimated Conv3x3 time: %.6f ms  "
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
      q.parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                     [=](sycl::nd_item<1> item) {
                       heun_predict(current, x0_cur, d_prime, elements, s, item);
                     });
      if (s.correct) {
        // The second denoiser call reuses its output allocation.
        q.memcpy(d_x0_cur, x0_cur, bytes);
        if (denoiser(d_prime, s.t_next) != 0) return 1;
        float* x0_prime = denoiser.output();
        q.parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                       [=](sycl::nd_item<1> item) {
                         heun_correct(current, d_x0_cur, x0_prime, next, elements,
                                      s, item);
                       });
      } else {
        q.parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                       [=](sycl::nd_item<1> item) {
                         heun_correct(current, x0_cur, x0_cur, next, elements, s,
                                      item);
                       });
      }
      std::swap(current, next);
    }
    return 0;
  };

  // Warm-up
  q.memcpy(d_a, host_x.data(), bytes).wait();
  for (int r = 0; r < 100; ++r) if (sample() != 0) return 1;
  q.wait();

  double seconds = 0;
  for (int r = 0; r < repeat; ++r) {
    q.memcpy(d_a, host_x.data(), bytes).wait();
    const auto start = std::chrono::steady_clock::now();
    if (sample() != 0) return 1;
    q.wait();
    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              start).count();
  }

  // sample() writes d_b, d_a, ... and swaps once per step, so an even step
  // count leaves the latent in d_a and an odd count leaves it in d_b.
  const float* gpu_final = (steps % 2 == 0) ? d_a : d_b;
  std::vector<float> output(elements);
  q.memcpy(output.data(), gpu_final, output.size() * sizeof(float)).wait();
  bool finite = true;
  double sum = 0.0;
  double sum_sq = 0.0;
  for (float x : output) {
    finite &= std::isfinite(x);
    sum += x;
    sum_sq += static_cast<double>(x) * x;
  }
  printf("Average sampling time: %.6f s (%d denoiser calls: 2x oneDNN 3x3 conv)\n",
         seconds / repeat, 2 * steps - 1);
  // Direct convolution and the selected oneDNN algorithm accumulate in
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

  sycl::free(d_x0_cur, q);
  sycl::free(d_prime, q);
  sycl::free(d_b, q);
  sycl::free(d_a, q);
  return pass ? 0 : 1;
}
