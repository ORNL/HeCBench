#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstddef>
#include <sycl/sycl.hpp>
#include "kernels.h"
#include "timer.h"

// grids and blocks are constant for the findPeak kernel
#define findPeakNBlocks 128
#define findPeakWidth 256

struct Peak {
  size_t pos;
  float val;
};

struct Position {
    Position(int _x, int _y) : x(_x), y(_y) { };
  int x;
  int y;
};

static Position idxToPos(const size_t idx, const int width)
{
  const int y = idx / width;
  const int x = idx % width;
  return Position(x, y);
}

static size_t posToIdx(const int width, const Position& pos)
{
  return (pos.y * width) + pos.x;
}

static Peak findPeak(sycl::queue &q, float *d_image, Peak *d_peak, size_t size)
{
  const int nBlocks = findPeakNBlocks;
  Peak peaks[nBlocks];

  // Initialise a peaks array on the device. Each thread block will return
  // a peak. Note:  the d_peaks array is not initialized (hence avoiding the
  // memcpy), it is up to the device function to do that

  sycl::range<1> gws (nBlocks * findPeakWidth);
  sycl::range<1> lws (findPeakWidth);

  // Find peak
  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<float, 1> maxVal(sycl::range<1>(findPeakWidth), cgh);
    sycl::local_accessor<size_t, 1> maxPos(sycl::range<1>(findPeakWidth), cgh);
    cgh.parallel_for<class find_peak>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int tid = item.get_local_id(0);
      int bid = item.get_group(0);
      float val = 0.f;
      size_t pos = 0;

      for (size_t idx = item.get_global_id(0);
                  idx < size; idx += findPeakWidth*findPeakNBlocks) {
        if (sycl::fabs(d_image[idx]) > sycl::fabs(val)) {
          val = d_image[idx];
          pos = idx;
        }
      }
      maxVal[tid] = val;
      maxPos[tid] = pos;

      item.barrier(sycl::access::fence_space::local_space);

      if (tid == 0) {
        val = 0.f, pos = 0;
        for (int i = 0; i < findPeakWidth; ++i) {
          if (sycl::fabs(maxVal[i]) > sycl::fabs(val)) {
            val = maxVal[i];
            pos = maxPos[i];
          }
        }
        d_peak[bid].val = val;
        d_peak[bid].pos = pos;
      }
    });
  });

  // Get the peaks array back from the device
  q.memcpy(&peaks, d_peak, nBlocks * sizeof(Peak)).wait();

  // Each thread block returned a peak, find the absolute maximum
  Peak p;
  p.val = 0.f;
  p.pos = 0;
  for (int i = 0; i < nBlocks; ++i) {
    if (fabsf(peaks[i].val) > fabsf(p.val)) {
      p.val = peaks[i].val;
      p.pos = peaks[i].pos;
    }
  }

  return p;
}

static void subtractPSF(
  sycl::queue &q,
  float *d_psf, const int psfWidth,
  float *d_residual, const int residualWidth,
  const size_t peakPos, const size_t psfPeakPos,
  const float absPeakVal, const float gain)
{
  const int blockDim = 16;

  const int rx = idxToPos(peakPos, residualWidth).x;
  const int ry = idxToPos(peakPos, residualWidth).y;

  const int px = idxToPos(psfPeakPos, psfWidth).x;
  const int py = idxToPos(psfPeakPos, psfWidth).y;

  const int diffx = rx - px;
  const int diffy = ry - px;

  const int startx = std::max(0, rx - px);
  const int starty = std::max(0, ry - py);

  const int stopx = std::min(residualWidth - 1, rx + (psfWidth - px - 1));
  const int stopy = std::min(residualWidth - 1, ry + (psfWidth - py - 1));

  // Note: Both start* and stop* locations are inclusive.
  const int blocksx = (stopx-startx + blockDim) / blockDim;
  const int blocksy = (stopy-starty + blockDim) / blockDim;

  sycl::range<2> gws (blocksy * blockDim, blocksx * blockDim);
  sycl::range<2> lws (blockDim, blockDim);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class subtract_PSF>(
      sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      const int x = startx + item.get_global_id(1);
      const int y = starty + item.get_global_id(0);

      // thread blocks are of size 16, but the workload is not always a multiple of 16
      if (x <= stopx && y <= stopy) {
        d_residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
          * d_psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))];
      }
    });
  });
}

HogbomTest::HogbomTest()
{
}

HogbomTest::~HogbomTest()
{
}

void HogbomTest::deconvolve(const std::vector<float>& dirty,
    const size_t dirtyWidth,
    const std::vector<float>& psf,
    const size_t psfWidth,
    std::vector<float>& model,
    std::vector<float>& residual)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  residual = dirty;

  // Initialise a peaks array on the device. Each thread block will return
  // a peak. Note:  the d_peaks array is not initialized (hence avoiding the
  // memcpy), it is up to the device function to do that
  Peak *d_peaks = sycl::malloc_device<Peak>(findPeakNBlocks, q);

  const size_t psf_size = psf.size();
  const size_t residual_size = residual.size();
  float *d_psf = sycl::malloc_device<float>(psf_size, q);
  q.memcpy(d_psf, &psf[0], psf_size * sizeof(float));

  float *d_residual = sycl::malloc_device<float>(residual_size, q);
  q.memcpy(d_residual, &residual[0], residual_size * sizeof(float));

  // Find peak of PSF
  Peak psfPeak = findPeak(q, d_psf, d_peaks, psf_size);

  q.wait();
  Stopwatch sw;
  sw.start();

  std::cout << "Found peak of PSF: " << "Maximum = " << psfPeak.val 
    << " at location " << idxToPos(psfPeak.pos, psfWidth).x << ","
    << idxToPos(psfPeak.pos, psfWidth).y << std::endl;
  assert(psfPeak.pos <= psf_size);

  for (unsigned int i = 0; i < niters; ++i) {
    // Find peak in the residual image
    Peak peak = findPeak(q, d_residual, d_peaks, residual_size);

    assert(peak.pos <= residual_size);

    // Check if threshold has been reached
    if (fabsf(peak.val) < threshold) {
      std::cout << "Reached stopping threshold" << std::endl;
    }

    // Subtract the PSF from the residual image 
    subtractPSF(q, d_psf, psfWidth, d_residual, dirtyWidth, peak.pos, psfPeak.pos, peak.val, gain);

    // Add to model
    model[peak.pos] += peak.val * gain;
  }

  q.wait();
  const double time = sw.stop();

  // Report on timings
  std::cout << "    Time " << time << " (s) " << std::endl;
  std::cout << "    Time per cycle " << time / niters * 1000 << " (ms)" << std::endl;
  std::cout << "    Cleaning rate  " << niters / time << " (iterations per second)" << std::endl;
  std::cout << "Done" << std::endl;

  // Copy device arrays back into the host 
  q.memcpy(&residual[0], d_residual, residual.size() * sizeof(float)).wait();

  // Free device memory
  sycl::free(d_peaks, q);
  sycl::free(d_psf, q);
  sycl::free(d_residual, q);
}
