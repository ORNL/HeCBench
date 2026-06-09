// Copyright (c) 2016 Nicolas Weber and Sandra C. Amend / GCC / TU-Darmstadt.
// All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include "shared.h"
#include "reference.h"

double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}

void run(const Params& i, const uchar3* hInput, uchar3* hOutput);

int main(int argc, char** argv) {
  // generate a random image for testing
  if(argc != 5) {
    printf("Usage: %s <output image width> <output image height> ", argv[0]);
    // explore the influence of lambda 
    printf("<lambda> <repeat>\n");
    exit(1);
  }

  // read params
  Params p;
  p.oWidth = (uint32_t)std::atoi(argv[1]);
  p.oHeight = (uint32_t)std::atoi(argv[2]);
  p.lambda = (float)std::atof(argv[3]);
  p.repeat = (uint32_t)std::atoi(argv[4]);

  // check params
  if(p.oWidth == 0 && p.oHeight == 0) {
    printf("only one dimension (width or height) can be 0!\n");
    exit(1);
  }

  p.iWidth  = 8192;
  p.iHeight = 8192;

  // calc width/height according to aspect ratio
  if(p.oWidth == 0)
    p.oWidth = (uint32_t)std::round((p.oHeight / (double)p.iHeight) * p.iWidth);

  if(p.oHeight == 0)
    p.oHeight = (uint32_t)std::round((p.oWidth / (double)p.iWidth) * p.iHeight);

  // calc patch size
  p.pWidth  = p.iWidth  / (float)p.oWidth;
  p.pHeight  = p.iHeight / (float)p.oHeight;

  // set random values for an image
  uchar3 *hInput   = (uchar3*) malloc (sizeof(uchar3) * p.iWidth * p.iHeight);
  uchar3 *hOutput  = (uchar3*) malloc (sizeof(uchar3) * p.oWidth * p.oHeight);

  uint64_t seed = 123;
  for (uint32_t i = 0; i < p.iWidth * p.iHeight; i++) {
    hInput[i][0] = (unsigned char)(256*LCG_random_double(&seed));
    hInput[i][1] = (unsigned char)(256*LCG_random_double(&seed));
    hInput[i][2] = (unsigned char)(256*LCG_random_double(&seed));
  }
  
  // run downsampling on a device
  run(p, hInput, hOutput);

  // per-pixel comparison for validation
  uchar3 *hRef = (uchar3*) malloc(sizeof(uchar3) * p.oWidth * p.oHeight);
  reference(p.oWidth, p.oHeight, p.iWidth, p.iHeight,
            p.pWidth, p.pHeight, p.lambda,
            hInput, hRef);
  uint32_t mismatches = 0;
  int maxDiff = 0;
  for (uint32_t i = 0; i < p.oWidth * p.oHeight; i++) {
    int dx = std::abs((int)hOutput[i][0] - (int)hRef[i][0]);
    int dy = std::abs((int)hOutput[i][1] - (int)hRef[i][1]);
    int dz = std::abs((int)hOutput[i][2] - (int)hRef[i][2]);
    if (dx > 0 || dy > 0 || dz > 0) mismatches++;
    if (dx > maxDiff) maxDiff = dx;
    if (dy > maxDiff) maxDiff = dy;
    if (dz > maxDiff) maxDiff = dz;
  }
  uint32_t total = p.oWidth * p.oHeight;
  if (mismatches == 0) {
    printf("Verification PASS: GPU matches CPU reference exactly.\n");
  } else {
    printf("Verification: %u / %u pixels differ (%.2f%%), max channel diff = %d\n",
           mismatches, total, 100.0 * mismatches / total, maxDiff);
    if (maxDiff <= 1)
      printf("  -> within rounding tolerance (maxDiff<=1, likely OK)\n");
    else
      printf("  -> WARNING: differences exceed rounding tolerance\n");
  }

  free(hRef);
  free(hInput);
  free(hOutput);
  return 0;
}
