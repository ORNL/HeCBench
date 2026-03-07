#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <sys/stat.h>

#include "utils.h"
#include "timer.h"
#include "reference.h"
#include "kernels.h"

unsigned int niters;
const float gain = 0.1f;
const float threshold = 0.00001f;

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::cout << "Usage: " << argv[0]
              << " <dirty image> <PSF file> <repeat>" << std::endl;
    return 1;
  }

  // Load dirty image and psf
  const char* dirtyFile = argv[1];  // dirty.img
  const char* psfFile = argv[2];    // psf.img
  niters = atoi(argv[3]);           // iterations

  std::cout << "Reading dirty image and psf image" << std::endl;
  std::vector<float> dirty = readImage(dirtyFile);
  const size_t dim = checkSquare(dirty);

  std::vector<float> psf = readImage(psfFile);
  const size_t psfDim = checkSquare(psf);

  // Reports some numbers
  std::cout << "Iterations = " << niters << std::endl;
  std::cout << "Image dimensions = " << dim << "x" << dim << std::endl;

#ifdef VERIFY
  // Run the golden version of the code
  std::vector<float> goldenResidual;
  std::vector<float> goldenModel(dirty.size());
  zeroInit(goldenModel);
  {
    // Time the serial (Golden) CPU implementation (may take a while)
    std::cout << "+++++ Serial processing on a CPU +++++" << std::endl;
    HogbomGolden golden;

    Stopwatch sw;
    sw.start();
    golden.deconvolve(dirty, dim, psf, psfDim, goldenModel, goldenResidual);
    const double time = sw.stop();

    // Report timings
    std::cout << "    Time " << time << " (s) " << std::endl;
    std::cout << "    Time per cycle " << time / niters * 1000 << " (ms)" << std::endl;
    std::cout << "    Cleaning rate  " << niters / time << " (iterations per second)" << std::endl;
    std::cout << "Done" << std::endl;
  }
#endif

#ifdef OUTPUT
  // Write images out
  writeImage("residual.img", goldenResidual);
  writeImage("model.img", goldenModel);
#endif

  // Run the device version of the code
  std::vector<float> deviceResidual;
  std::vector<float> deviceModel(dirty.size());
  zeroInit(deviceModel);
  {
    // Time for the implementation targeting a device
    std::cout << "+++++ Offload processing on a device +++++" << std::endl;
    HogbomTest device;
    device.deconvolve(dirty, dim, psf, psfDim, deviceModel, deviceResidual);
  }

#ifdef VERIFY
  std::cout << "Verifying model...";
  const bool modelDiff = compare(goldenModel, deviceModel);
  if (!modelDiff) {
    std::cout << "FAIL" << std::endl;
    exit(1);
  } else {
    std::cout << "PASS" << std::endl;
  }

  std::cout << "Verifying residual...";
  const bool residualDiff = compare(goldenResidual, deviceResidual);
  if (!residualDiff) {
    std::cout << "FAIL" << std::endl;
    exit(1);
  } else {
    std::cout << "PASS" << std::endl;
  }
#endif

  return 0;
}
