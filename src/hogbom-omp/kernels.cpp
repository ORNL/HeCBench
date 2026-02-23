#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstddef>
#include <omp.h>
#include "kernels.h"
#include "timer.h"

// grids and blocks are constant for the findPeak kernel
#define findPeakNBlocks 128
#define findPeakWidth 256

struct Peak {
  size_t pos;
  float val;
};

#pragma omp declare target
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
#pragma omp end declare target

void k_findPeak(
  const float *__restrict image, 
  const size_t size,
  Peak *__restrict absPeak)
{

  #pragma omp target teams num_teams(findPeakNBlocks) thread_limit(findPeakWidth)
  {
    float maxVal[findPeakWidth];
    size_t maxPos[findPeakWidth];
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int bid = omp_get_team_num();
      float val = 0.f;
      size_t pos = 0;

      for (size_t idx = tid + bid * findPeakWidth;
                  idx < size; idx += findPeakWidth * findPeakNBlocks) {
        if (fabsf(image[idx]) > fabsf(val)) {
          val = image[idx];
          pos = idx;
        }
      }
      maxVal[tid] = val;
      maxPos[tid] = pos;

      #pragma omp barrier

      if (tid == 0) {
        val = 0.f, pos = 0;
        for (int i = 0; i < findPeakWidth; ++i) {
          if (fabsf(maxVal[i]) > fabsf(val)) {
            val = maxVal[i];
            pos = maxPos[i];
          }
        }
        absPeak[bid].val = val;
        absPeak[bid].pos = pos;
      }
    }
  }
}

static Peak findPeak(const float* d_image, Peak *d_peaks, size_t size)
{
  const int nBlocks = findPeakNBlocks;

  // Find peak
  k_findPeak(d_image, size, d_peaks);

  // Get the peaks array back from the device
  #pragma omp target update from (d_peaks[0:nBlocks])

  // Each thread block returned a peak, find the absolute maximum
  Peak p;
  p.val = 0.f;
  p.pos = 0;
  for (int i = 0; i < nBlocks; ++i) {
    if (fabsf(d_peaks[i].val) > fabsf(p.val)) {
      p.val = d_peaks[i].val;
      p.pos = d_peaks[i].pos;
    }
  }

  return p;
}

static void subtractPSF(const float* d_psf, const int psfWidth,
    float* d_residual, const int residualWidth,
    const size_t peakPos, const size_t psfPeakPos,
    const float absPeakVal, const float gain)
{
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

  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int y = starty; y <= stopy; y++) 
    for (int x = startx; x <= stopx; x++) 
    d_residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
      * d_psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))];
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

  Peak d_peaks[findPeakNBlocks];

  const size_t psf_size = psf.size();
  const size_t residual_size = residual.size();
  const float* d_psf = &psf[0];
  float* d_residual = &residual[0];

  #pragma omp target data map(to: d_psf[0:psf_size])\
                          map(tofrom: d_residual[0:residual_size]) \
                          map(alloc: d_peaks[0:findPeakNBlocks])
  {
    // Find peak of PSF
    Peak psfPeak = findPeak(d_psf, d_peaks, psf_size);

    std::cout << "Found peak of PSF: " << "Maximum = " << psfPeak.val 
      << " at location " << idxToPos(psfPeak.pos, psfWidth).x << ","
      << idxToPos(psfPeak.pos, psfWidth).y << std::endl;
    assert(psfPeak.pos <= psf_size);

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

    const double time = sw.stop();
    // Report on timings
    std::cout << "    Time " << time << " (s) " << std::endl;
    std::cout << "    Time per cycle " << time / niters * 1000 << " (ms)" << std::endl;
    std::cout << "    Cleaning rate  " << niters / time << " (iterations per second)" << std::endl;
    std::cout << "Done" << std::endl;
  }
}
