/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  •   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  •   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include "urng.h"
#include "kernel.cpp"

int main(int argc, char** argv) 
{
  if (argc != 3) {
    printf("Usage: %s <path to file> <repeat>\n", argv[0]);
    return 1;
  }
  const char* filePath = argv[1];
  const int iterations = atoi(argv[2]);

  // load input bitmap image
  SDKBitMap inputBitmap;   
  inputBitmap.load(filePath);
  if(!inputBitmap.isLoaded())
  {
    std::cout << "Failed to load input image!";
    return -1;
  }

  // get width and height of input image
  int height = inputBitmap.getHeight();
  int width = inputBitmap.getWidth();
  int size = height * width;
  size_t imageSize = size * sizeof(uchar4);

  std::cout << "Image " << filePath;
  std::cout << " height: " << height;
  std::cout << " width: " << width << std::endl;

  // allocate memory for input & output image data
  uchar4* inputImageData  = (uchar4*)malloc(imageSize);

  // allocate memory for output image data
  uchar4* outputImageData = (uchar4*)malloc(imageSize);

  // initializa the Image data to NULL
  memset(outputImageData, 0, imageSize);

  // get the pointer to pixel data
  uchar4 *pixelData = inputBitmap.getPixels();
  if(pixelData == NULL)
  {
    std::cout << "Failed to read pixel Data!";
    free(inputImageData);
    free(outputImageData);
    return -1;
  }

  // Copy pixel data into inputImageData
  memcpy(inputImageData, pixelData, imageSize);

  // allocate memory for verification output
  uchar4 *verificationOutput = (uchar4*)malloc(imageSize);

  // initialize the data to NULL
  memset(verificationOutput, 0, imageSize);

  const int factor = FACTOR;

  const int block = 256;
  const int teams = (size + 255) / block;

  std::cout << "Executing kernel for " << iterations << " iterations" <<std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  #pragma omp target data map(to: inputImageData[0:size]) \
                          map(from: outputImageData[0:size])
  {
    auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < iterations; i++)
    {
      #pragma omp target teams num_teams(teams) thread_limit(block)
      {
        int iv[NTAB * GROUP_SIZE];
        #pragma omp parallel 
        {
          int pos = omp_get_team_num() * block + omp_get_thread_num(); 
          if (pos < size) {
            float4 temp = convert_float4(inputImageData[pos]);

            /* compute average value of a pixel from its compoments */
            float avg = (temp.x + temp.y + temp.z + temp.w) / 4.0f;

            /* Calculate deviation from the avg value of a pixel */
            float dev = ran1(-avg, iv);
            dev = (dev - 0.55f) * (float)factor;

            /* Saturate(clamp) the values */
            outputImageData[pos] = convert_uchar4_sat(temp + dev);
          }
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time: " <<  (time * 1e-3f) / iterations << " (us)\n";
  }

  // verify
  float mean = 0;
  for(int i = 0; i < size; i++)
  {
    mean += outputImageData[i].x - inputImageData[i].x;
    mean += outputImageData[i].y - inputImageData[i].y;
    mean += outputImageData[i].z - inputImageData[i].z;
    mean += outputImageData[i].w - inputImageData[i].w;
  }

  mean /= (imageSize * factor);
  std::cout << "The averaged mean of the image: " << mean << std::endl;

  if(fabs(mean) < 1.0)
  {
    std::cout << "PASS" << std::endl;
  }
  else
  {
    std::cout << "FAIL" << std::endl;
    exit(1);
  }

#ifdef DUMP
  // copy output image data back to original pixel data
  memcpy(pixelData, outputImageData, imageSize);

  // write the output bmp file
  if(!inputBitmap.write(OUTPUT_IMAGE))
    std::cout << "Failed to write output image!";
  else 
    std::cout << "Write output image!";
#endif

  // release program resources (input memory etc.)
  free(inputImageData);
  free(outputImageData);
  free(verificationOutput);
  return 0;
}


