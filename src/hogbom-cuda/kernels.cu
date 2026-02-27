#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstddef>
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
  __host__ __device__
    Position(int _x, int _y) : x(_x), y(_y) { };
  int x;
  int y;
};

__host__ __device__
static Position idxToPos(const size_t idx, const int width)
{
  const int y = idx / width;
  const int x = idx % width;
  return Position(x, y);
}

__device__
static size_t posToIdx(const int width, const Position& pos)
{
  return (pos.y * width) + pos.x;
}

__global__
void k_findPeak(
  const float *__restrict__ image, 
  const size_t size,
  Peak *__restrict__ absPeak)
{
  __shared__ float maxVal[findPeakWidth];
  __shared__ size_t maxPos[findPeakWidth];
  float val = 0.f;
  size_t pos = 0;

  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
              idx < size; idx += findPeakWidth * findPeakNBlocks) {
    if (fabsf(image[idx]) > fabsf(val)) {
      val = image[idx];
      pos = idx;
    }
  }
  maxVal[threadIdx.x] = val;
  maxPos[threadIdx.x] = pos;

  __syncthreads();

  if (threadIdx.x == 0) {
    val = 0.f, pos = 0;
    for (int i = 0; i < findPeakWidth; ++i) {
      if (fabsf(maxVal[i]) > fabsf(val)) {
        val = maxVal[i];
        pos = maxPos[i];
      }
    }
    absPeak[blockIdx.x].val = val;
    absPeak[blockIdx.x].pos = pos;
  }
}

static Peak findPeak(const float* d_image, Peak* d_peak, size_t size)
{
  const int nBlocks = findPeakNBlocks;
  Peak peaks[nBlocks];

  // Find peak
  k_findPeak<<<nBlocks, findPeakWidth>>>(d_image, size, d_peak);

  // Get the peaks array back from the device
  cudaMemcpy(&peaks, d_peak, nBlocks * sizeof(Peak), cudaMemcpyDeviceToHost);

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

__global__
void k_subtractPSF(
    const float *__restrict__ d_psf,
    const int psfWidth,
          float *__restrict__ d_residual,
    const int residualWidth,
    const int startx, const int starty,
    const int stopx, const int stopy,
    const int diffx, const int diffy,
    const float absPeakVal, const float gain)
{   
  const int x = startx + threadIdx.x + blockIdx.x * blockDim.x;
  const int y = starty + threadIdx.y + blockIdx.y * blockDim.y;

  // thread blocks are of size 16, but the workload is not always a multiple of 16
  if (x <= stopx && y <= stopy) {
    d_residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
      * d_psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))];
  }
}

static void subtractPSF(const float* d_psf, const int psfWidth,
    float* d_residual, const int residualWidth,
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

  dim3 numBlocks(blocksx, blocksy);
  dim3 threadsPerBlock(blockDim, blockDim);
  k_subtractPSF<<<numBlocks,threadsPerBlock>>>(d_psf, psfWidth, d_residual, residualWidth,
      startx, starty, stopx, stopy, diffx, diffy, absPeakVal, gain);
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
  residual = dirty;

  // Initialise a peaks array on the device. Each thread block will return
  // a peak. Note:  the d_peaks array is not initialized (hence avoiding the
  // memcpy), it is up to the device function to do that
  Peak* d_peaks;
  cudaMalloc((void **) &d_peaks, findPeakNBlocks * sizeof(Peak));

  float* d_psf;
  float* d_residual;
  const size_t psf_size = psf.size();
  const size_t residual_size = residual.size();
  cudaMalloc((void **) &d_psf, psf_size * sizeof(float));
  cudaMalloc((void **) &d_residual, residual_size * sizeof(float));

  cudaMemcpy(d_psf, &psf[0], psf_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_residual, &residual[0], residual_size * sizeof(float), cudaMemcpyHostToDevice);

  // Find peak of PSF
  Peak psfPeak = findPeak(d_psf, d_peaks, psf_size);

  std::cout << "Found peak of PSF: " << "Maximum = " << psfPeak.val 
    << " at location " << idxToPos(psfPeak.pos, psfWidth).x << ","
    << idxToPos(psfPeak.pos, psfWidth).y << std::endl;
  assert(psfPeak.pos <= psf_size);

  cudaDeviceSynchronize();
  Stopwatch sw;
  sw.start();

  for (unsigned int i = 0; i < niters; ++i) {
    // Find peak in the residual image
    Peak peak = findPeak(d_residual, d_peaks, residual_size);

    assert(peak.pos <= residual_size);

    // Check if threshold has been reached
    if (fabsf(peak.val) < threshold) {
      std::cout << "Reached stopping threshold" << std::endl;
    }

    // Subtract the PSF from the residual image (this function will launch
    // an kernel asynchronously, need to sync later
    subtractPSF(d_psf, psfWidth, d_residual, dirtyWidth, peak.pos, psfPeak.pos, peak.val, gain);

    // Add to model
    model[peak.pos] += peak.val * gain;
  }

  cudaDeviceSynchronize();
  const double time = sw.stop();

  // Report on timings
  std::cout << "    Time " << time << " (s) " << std::endl;
  std::cout << "    Time per cycle " << time / niters * 1000 << " (ms)" << std::endl;
  std::cout << "    Cleaning rate  " << niters / time << " (iterations per second)" << std::endl;
  std::cout << "Done" << std::endl;

  // Copy device arrays back into the host 
  cudaMemcpy(&residual[0], d_residual, residual.size() * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_peaks);
  cudaFree(d_psf);
  cudaFree(d_residual);
}
