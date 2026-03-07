/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// *********************************************************************
// Demo application for realtime DXT1 compression based on the OpenCL
// DXTC sample
// *********************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <omp.h>
#include "dds.h"
#include "permutations.h"
#include "block.h"
#include "shrUtils.h"

#define ERROR_THRESHOLD 0.02f
#define NUM_THREADS     64      // Number of threads per work group.

typedef struct { float x; float y; float z; float w; } float4 ;
typedef struct { uint x; uint y; uint z; uint w;} uint4 ;
#include "kernel.cpp"


// Main function
// *********************************************************************
int main(int argc, char** argv) 
{
  if (argc != 4) {
    printf("Usage: %s <path to image> <path to reference image> <repeat>\n", argv[0]);
    return 1;
  }
  const char* image_path = argv[1];
  assert(image_path != NULL);

  const char* reference_image_path = argv[2];
  assert(reference_image_path != NULL);

  const int numIterations = atoi(argv[3]);

  unsigned int width, height;
  unsigned int* h_img = NULL;
  const float alphaTable4[4] = {9.0f, 0.0f, 6.0f, 3.0f};
  const float alphaTable3[4] = {4.0f, 0.0f, 2.0f, 2.0f};
  const int prods4[4] = {0x090000, 0x000900, 0x040102, 0x010402};
  const int prods3[4] = {0x040000, 0x000400, 0x040101, 0x010401};

  // load image 
  if (!shrLoadPPM4ub(image_path, (unsigned char **)&h_img, &width, &height)) {
    printf("Error, unable to open source image file <%s>\n", image_path);

    exit(EXIT_FAILURE);
  }

  if (width % 4 != 0 || height % 4 != 0) {
    printf("Error: the image dimensions must be a multiple of 4.\n");
    free(h_img);
    exit(EXIT_FAILURE);
  }

  if (width > 16384 || height > 16384) {
    printf("Error: the image dimensions exceed the maximum values.\n");
    free(h_img);
    exit(EXIT_FAILURE);
  }

  printf("Loaded '%s', %d x %d pixels\n\n", image_path, width, height);

  // Convert linear image to block linear. 
  const unsigned int memSize = width * height;
  const unsigned int memSizeByte = memSize * sizeof(unsigned int);
  unsigned int* block_image = (unsigned int*)malloc(memSizeByte);

  // Convert linear image to block linear. 
  for(unsigned int by = 0; by < height/4; by++) {
    for(unsigned int bx = 0; bx < width/4; bx++) {
      for (int i = 0; i < 16; i++) {
        const int x = i & 3;
        const int y = i / 4;
        block_image[(by * width/4 + bx) * 16 + i] = 
          ((unsigned int *)h_img)[(by * 4 + y) * 4 * (width/4) + bx * 4 + x];
      }
    }
  }

  // Compute permutations.
  unsigned int permutations[1024];
  computePermutations(permutations);

  const unsigned int compressedSize = (width / 4) * (height / 4) * 8;
  unsigned int * h_result = (unsigned int*)malloc(compressedSize);

  #pragma omp target data map(to: permutations[0:1024], \
                                  prods4[0:4], \
                                  prods3[0:4], \
                                  alphaTable4[0:4], \
                                  alphaTable3[0:4], \
                                  block_image[0:memSize]) \
                          map(from: h_result[0:compressedSize/4])
{
  // Determine launch configuration and run timed computation numIterations times
  int blocks = width / 4 * height * 4;

  // Restrict the numbers of blocks to launch on low end GPUs to avoid kernel timeout
  unsigned int compute_units = 24;
  int blocksPerLaunch = MIN(blocks, 768 * (int)compute_units);

  printf("\nRunning DXT Compression on %u x %u image...\n", width, height);
  printf("\n%u Workgroups, %u Work Items per Workgroup, %u Work Items in NDRange...\n\n", 
      blocks, NUM_THREADS, blocks * NUM_THREADS);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numIterations; ++i) {
    for( int j=0; j<blocks; j+= blocksPerLaunch ) {
      int grid = MIN( blocksPerLaunch, blocks-j );
      #pragma omp target teams num_teams(grid) thread_limit(NUM_THREADS)
      {
        float4 colors[16];
        float4 sums[16];
        int s_int[64];
        float s_float[96];
        unsigned int s_permutations[160];
        int xrefs[16];
        #pragma omp parallel 
        {
          const int idx = omp_get_thread_num();

          loadColorBlock(block_image, colors, sums, xrefs, s_float, j);
          #pragma omp barrier

          uint4 best = evalAllPermutations(colors, permutations, s_float, sums[0], 
                                           s_permutations, alphaTable4, prods4, alphaTable3, prods3);

          // Use a parallel reduction to find minimum error.
          const int minIdx = findMinError(s_float, s_int);

          #pragma omp barrier

          // Only write the result of the winner thread.
          if (idx == minIdx)
          {
            saveBlockDXT1(best.x, best.y, best.z, xrefs, h_result, j);
          }
        }
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / numIterations);
}

  // Write DDS file.
  FILE* fp = NULL;
  char output_filename[1024];
#ifdef WIN32
  strcpy_s(output_filename, 1024, image_path);
  strcpy_s(output_filename + strlen(image_path) - 3, 1024 - strlen(image_path) + 3, "dds");
  fopen_s(&fp, output_filename, "wb");
#else
  strcpy(output_filename, image_path);
  strcpy(output_filename + strlen(image_path) - 3, "dds");
  fp = fopen(output_filename, "wb");
#endif
  assert(fp != NULL);

  DDSHeader header;
  header.fourcc = FOURCC_DDS;
  header.size = 124;
  header.flags  = (DDSD_WIDTH|DDSD_HEIGHT|DDSD_CAPS|DDSD_PIXELFORMAT|DDSD_LINEARSIZE);
  header.height = height;
  header.width = width;
  header.pitch = compressedSize;
  header.depth = 0;
  header.mipmapcount = 0;
  memset(header.reserved, 0, sizeof(header.reserved));
  header.pf.size = 32;
  header.pf.flags = DDPF_FOURCC;
  header.pf.fourcc = FOURCC_DXT1;
  header.pf.bitcount = 0;
  header.pf.rmask = 0;
  header.pf.gmask = 0;
  header.pf.bmask = 0;
  header.pf.amask = 0;
  header.caps.caps1 = DDSCAPS_TEXTURE;
  header.caps.caps2 = 0;
  header.caps.caps3 = 0;
  header.caps.caps4 = 0;
  header.notused = 0;

  fwrite(&header, sizeof(DDSHeader), 1, fp);
  fwrite(h_result, compressedSize, 1, fp);

  fclose(fp);

  // Make sure the generated image matches the reference image (regression check)
  printf("\nComparing against Host/C++ computation...\n");     

  // read in the reference image from file
#ifdef WIN32
  fopen_s(&fp, reference_image_path, "rb");
#else
  fp = fopen(reference_image_path, "rb");
#endif
  assert (fp != NULL);
  fseek(fp, sizeof(DDSHeader), SEEK_SET);
  unsigned int referenceSize = (width / 4) * (height / 4) * 8;
  unsigned int * reference = (unsigned int *)malloc(referenceSize);
  fread(reference, referenceSize, 1, fp);
  fclose(fp);

  // compare the reference image data to the sample/generated image
  float rms = 0;
  for (unsigned int y = 0; y < height; y += 4)
  {
    for (unsigned int x = 0; x < width; x += 4)
    {
      // binary comparison of data
      unsigned int referenceBlockIdx = ((y/4) * (width/4) + (x/4));
      unsigned int resultBlockIdx = ((y/4) * (width/4) + (x/4));
      int cmp = compareBlock(((BlockDXT1 *)h_result) + resultBlockIdx, ((BlockDXT1 *)reference) + referenceBlockIdx);

      // log deviations, if any
      if (cmp != 0.0f) 
      {
        compareBlock(((BlockDXT1 *)h_result) + resultBlockIdx, ((BlockDXT1 *)reference) + referenceBlockIdx);
        //printf("Deviation at (%d, %d):\t%f rms\n", x/4, y/4, float(cmp)/16/3);
      }
      rms += cmp;
    }
  }
  rms /= width * height * 3;
  printf("RMS(reference, result) = %f\n\n", rms);

  // Free host memory
  free(block_image);
  free(h_result);
  free(h_img);
  free(reference);

  // finish
  if (rms <= ERROR_THRESHOLD) 
    printf("PASS\n"); 
  else
    printf("FAIL\n");
  if (rms > ERROR_THRESHOLD) exit(1);

  return 0;
}
