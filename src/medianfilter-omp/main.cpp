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

#include <chrono>
#include <omp.h>
#include "shrUtils.h"
#include <cstdlib>

typedef struct __attribute__((__aligned__(4)))
{
  unsigned char x;
  unsigned char y;
  unsigned char z;
  unsigned char w;
} uchar4;


#ifndef min
#define min(a,b) (a < b ? a : b)
#endif

// Import host computation function
extern "C" void MedianFilterHost(unsigned int* uiInputImage, unsigned int* uiOutputImage,
                                 unsigned int uiWidth, unsigned int uiHeight);

double MedianFilterGPU(
    uchar4* uiInputImage,
    unsigned int* uiOutputImage,
    const int uiImageWidth,
    const int uiImageHeight);

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <image file> <repeat>\n", argv[0]);
    return 1;
  }
  // Image data file
  const char* cPathAndName = argv[1];

  const int iCycles = atoi(argv[2]);

  const unsigned int uiMaxImageWidth = 1920;   // Image width
  const unsigned int uiMaxImageHeight = 1080;  // Image height

  unsigned int uiImageWidth;
  unsigned int uiImageHeight;

  size_t szBuffBytes;                 // Size of main image buffers
  size_t szBuffWords;

  unsigned int* uiInput;              // Host input buffer
  unsigned int* uiOutput;             // Host output buffer

  // One device processes the whole image
  szBuffWords = uiMaxImageHeight * uiMaxImageWidth;
  szBuffBytes = szBuffWords * sizeof (unsigned int);

  uiInput = (unsigned int*) malloc (szBuffBytes);
  uchar4* uc4Source = (uchar4*) uiInput;

  bool status = shrLoadPPM4ub(cPathAndName, (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);

  printf("Image File\t = %s\nImage Dimensions = %u w x %u h x %lu bpp\n\n",
         cPathAndName, uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3);

  if (uiImageWidth > uiMaxImageWidth || uiImageHeight > uiMaxImageHeight) {
    printf("Error: Image Dimensions exceed the maximum values");
    status = 0;
  }
  if (!status) {
     free(uiInput);
     return 1;
  }

  uiOutput = (unsigned int*) malloc (szBuffBytes);

  #pragma omp target data map(to: uc4Source[0:szBuffWords])\
                          map(from: uiOutput[0:szBuffWords])
  {
    // Warmup call
    MedianFilterGPU (uc4Source, uiOutput, uiImageWidth, uiImageHeight);

    double time = 0.0;

    // Process n loops on the GPU
    printf("\nRunning MedianFilterGPU for %d cycles...\n\n", iCycles);
    for (int i = 0; i < iCycles; i++)
    {
      time += MedianFilterGPU (uc4Source, uiOutput, uiImageWidth, uiImageHeight);
    }
    printf("Average kernel execution time: %f (s)\n\n", (time * 1e-9f) / iCycles);
  }

  // Compute on host
  unsigned int* uiGolden = (unsigned int*)malloc(szBuffBytes);
  MedianFilterHost(uiInput, uiGolden, uiImageWidth, uiImageHeight);

  // Compare GPU and Host results:  Allow variance of 1 GV in up to 0.01% of pixels
  printf("Comparing GPU Result to CPU Result...\n");
  shrBOOL bMatch = shrCompareuit(uiGolden, uiOutput, szBuffWords, 1.0f, 0.0001f);
  printf("\nGPU Result %s CPU Result within tolerance...\n",
         (bMatch == shrTRUE) ? "matches" : "DOESN'T match");

  // Cleanup and exit
  free(uiGolden);
  free(uiInput);
  free(uiOutput);

  if(bMatch == shrTRUE)
    printf("PASS\n");
  else
    printf("FAIL\n");
  if (bMatch != shrTRUE) exit(1);

  return EXIT_SUCCESS;
}

// Copies input data from host buf to the device, runs kernel,
// copies output data back to output host buf
double MedianFilterGPU(
    uchar4* uc4Source,
    unsigned int* uiDest,
    const int iImageWidth,
    const int iImageHeight)
{
  size_t szGlobalWorkSize[2];         // 2D global work items (ND range) for Median kernel
  size_t szLocalWorkSize[2];          // 2D local work items (work group) for Median kernel
  const int iBlockDimX = 16;
  const int iBlockDimY = 4;
  const int iLocalPixPitch = iBlockDimX + 2;

  szLocalWorkSize[0] = iBlockDimX;
  szLocalWorkSize[1] = iBlockDimY;
  szGlobalWorkSize[0] = shrRoundUp((int)szLocalWorkSize[0], iImageWidth);
  szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], iImageHeight);

  auto start = std::chrono::steady_clock::now();

  int iTeamX = szGlobalWorkSize[0] / szLocalWorkSize[0];
  int iTeamY = szGlobalWorkSize[1] / szLocalWorkSize[1];

  int iNumTeams = iTeamX * iTeamY;
  int iNumThreads = iBlockDimX * iBlockDimY;

  #pragma omp target teams num_teams(iNumTeams) thread_limit(iNumThreads)
  {
    uchar4  uc4LocalData[iLocalPixPitch*(iBlockDimY+2)];
    #pragma omp parallel
    {
     // Get parent image x and y pixel coordinates from global ID,
     // and compute offset into parent GMEM data
     int iLocalIdX = omp_get_thread_num() % iBlockDimX;
     int iLocalIdY = omp_get_thread_num() / iBlockDimX;
     int iGroupIdX = omp_get_team_num() % iTeamX;
     int iGroupIdY = omp_get_team_num() / iTeamX;
     int iBlockX = iBlockDimX;
     int iBlockY = iBlockDimY;
     int iImagePosX = iGroupIdX * iBlockX + iLocalIdX;
     int iDevYPrime = iGroupIdY * iBlockY + iLocalIdY - 1;  // Shift offset up 1 radius (1 row) for reads
     int iImageX = iImageWidth;

     int iDevGMEMOffset = iDevYPrime * iImageX + iImagePosX;

     // Compute initial offset of current pixel within work group LMEM block
     int iLocalPixOffset = iLocalIdY * iLocalPixPitch + iLocalIdX + 1;

     // Main read of GMEM data into LMEM
     if((iDevYPrime > -1) && (iDevYPrime < iImageHeight) && (iImagePosX < iImageWidth))
     {
       uc4LocalData[iLocalPixOffset] = uc4Source[iDevGMEMOffset];
     }
     else
     {
       uc4LocalData[iLocalPixOffset] = {0,0,0,0};
     }

     // Work items with y ID < 2 read bottom 2 rows of LMEM
     if (iLocalIdY < 2)
     {
       // Increase local offset by 1 workgroup LMEM block height
       // to read in top rows from the next block region down
       iLocalPixOffset += iBlockY * iLocalPixPitch;

       // If source offset is within the image boundaries
       if (((iDevYPrime + iBlockY) < iImageHeight) && (iImagePosX < iImageWidth))
       {
         // Read in top rows from the next block region down
         uc4LocalData[iLocalPixOffset] = uc4Source[iDevGMEMOffset + iBlockY * iImageX];
       }
       else
       {
         uc4LocalData[iLocalPixOffset] = {0,0,0,0};
       }
     }

     // Work items with x ID at right workgroup edge will read Left apron pixel
     if (iLocalIdX == (iBlockX - 1))
     {
       // set local offset to read data from the next region over
       iLocalPixOffset = iLocalIdY * iLocalPixPitch;

       // If source offset is within the image boundaries and not at the leftmost workgroup
       if ((iDevYPrime > -1) && (iDevYPrime < iImageHeight) && (iGroupIdX > 0))
       {
         // Read data into the LMEM apron from the GMEM at the left edge of the next block region over
         uc4LocalData[iLocalPixOffset] = uc4Source[iDevYPrime * iImageX + iGroupIdX * iBlockX - 1];
       }
       else
       {
         uc4LocalData[iLocalPixOffset] = {0,0,0,0};
       }

       // If in the bottom 2 rows of workgroup block
       if (iLocalIdY < 2)
       {
         // Increase local offset by 1 workgroup LMEM block height
         // to read in top rows from the next block region down
         iLocalPixOffset += iBlockY * iLocalPixPitch;

         // If source offset in the next block down isn't off the image and not at the leftmost workgroup
         if (((iDevYPrime + iBlockY) < iImageHeight) && (iGroupIdX > 0))
         {
           // read in from GMEM (reaching down 1 workgroup LMEM block height and left 1 pixel)
           uc4LocalData[iLocalPixOffset] = uc4Source[(iDevYPrime + iBlockY) * iImageX +
		   iGroupIdX * iBlockX - 1];
         }
         else
         {
           uc4LocalData[iLocalPixOffset] = {0,0,0,0};
         }
       }
     }
     else if (iLocalIdX == 0) // Work items with x ID at left workgroup edge will read right apron pixel
     {
       // set local offset
       iLocalPixOffset = (iLocalIdY + 1) * iLocalPixPitch - 1;

       if ((iDevYPrime > -1) && (iDevYPrime < iImageHeight) &&
           ((iGroupIdX + 1) * iBlockX < iImageWidth))
       {
         // read in from GMEM (reaching left 1 pixel) if source offset is within image boundaries
         uc4LocalData[iLocalPixOffset] = uc4Source[iDevYPrime * iImageX + (iGroupIdX + 1) * iBlockX];
       }
       else
       {
         uc4LocalData[iLocalPixOffset] = {0,0,0,0};
       }

       // Read bottom 2 rows of workgroup LMEM block
       if (iLocalIdY < 2)
       {
         // increase local offset by 1 workgroup LMEM block height
         iLocalPixOffset += (iBlockY * iLocalPixPitch);

         if (((iDevYPrime + iBlockY) < iImageHeight) &&
             ((iGroupIdX + 1) * iBlockX < iImageWidth) )
         {
           // read in from GMEM (reaching down 1 workgroup LMEM block height and left 1 pixel) if source offset is within image boundaries
           uc4LocalData[iLocalPixOffset] = uc4Source[(iDevYPrime + iBlockY) * iImageX +
		   (iGroupIdX + 1) * iBlockX];
         }
         else
         {
           uc4LocalData[iLocalPixOffset] = {0,0,0,0};
         }
       }
     }

     // Synchronize the read into LMEM
     #pragma omp barrier

     // Compute
     // reset accumulators
     float fMedianEstimate[3] = {128.0f, 128.0f, 128.0f};
     float fMinBound[3] = {0.0f, 0.0f, 0.0f};
     float fMaxBound[3] = {255.0f, 255.0f, 255.0f};

     // now find the median using a binary search - Divide and Conquer 256 gv levels for 8 bit plane
     for(int iSearch = 0; iSearch < 8; iSearch++)  // for 8 bit data, use 0..8.  For 16 bit data, 0..16. More iterations for more bits.
     {
       unsigned int uiHighCount [3] = {0, 0, 0};

       // set local offset and kernel offset
       iLocalPixOffset = (iLocalIdY * iLocalPixPitch) + iLocalIdX;

       // Row1 Left Pix (RGB)
       uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);
       uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);
       uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);

       // Row1 Middle Pix (RGB)
       uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);
       uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);
       uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);

       // Row1 Right Pix (RGB)
       uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);
       uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);
       uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset].z);

       // set the offset into SMEM for next row
       iLocalPixOffset += (iLocalPixPitch - 2);

       // Row2 Left Pix (RGB)
       uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);
       uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);
       uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);

       // Row2 Middle Pix (RGB)
       uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);
       uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);
       uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);

       // Row2 Right Pix (RGB)
       uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);
       uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);
       uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset].z);

       // set the offset into SMEM for next row
       iLocalPixOffset += (iLocalPixPitch - 2);

       // Row3 Left Pix (RGB)
       uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);
       uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);
       uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);

       // Row3 Middle Pix (RGB)
       uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);
       uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);
       uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);

       // Row3 Right Pix (RGB)
       uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);
       uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);
       uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset].z);

       //********************************
       // reset the appropriate bound, depending upon counter
       if(uiHighCount[0] > 4)
       {
         fMinBound[0] = fMedianEstimate[0];
       }
       else
       {
         fMaxBound[0] = fMedianEstimate[0];
       }

       if(uiHighCount[1] > 4)
       {
         fMinBound[1] = fMedianEstimate[1];
       }
       else
       {
         fMaxBound[1] = fMedianEstimate[1];
       }

       if(uiHighCount[2] > 4)
       {
         fMinBound[2] = fMedianEstimate[2];
       }
       else
       {
         fMaxBound[2] = fMedianEstimate[2];
       }

       // refine the estimate
       fMedianEstimate[0] = 0.5f * (fMaxBound[0] + fMinBound[0]);
       fMedianEstimate[1] = 0.5f * (fMaxBound[1] + fMinBound[1]);
       fMedianEstimate[2] = 0.5f * (fMaxBound[2] + fMinBound[2]);
     }

     // pack into a monochrome unsigned int
     unsigned int uiPackedPix = 0x000000FF & (unsigned int)(fMedianEstimate[0] + 0.5f);
     uiPackedPix |= 0x0000FF00 & (((unsigned int)(fMedianEstimate[1] + 0.5f)) << 8);
     uiPackedPix |= 0x00FF0000 & (((unsigned int)(fMedianEstimate[2] + 0.5f)) << 16);

     // Write out to GMEM with restored offset
     if((iDevYPrime < iImageHeight) && (iImagePosX < iImageWidth))
     {
       uiDest[iDevGMEMOffset + iImageX] = uiPackedPix;
     }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return time;
}
