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


#include "sobel.h"
#include "SDKBitMap.h"
#include "kernels.cu"
#include <cstdlib>

static bool compare(const float *refData, const float *data,
                    const int length, const float epsilon = 1e-6f)
{
  float error = 0.0f;
  float ref = 0.0f;
  for(int i = 1; i < length; ++i)
  {
    float diff = refData[i] - data[i];
    // if (diff != 0) printf("mismatch @%d: %f %f\n", i, refData[i] , data[i]);
    error += diff * diff;
    ref += refData[i] * refData[i];
  }
  float normRef = sqrtf((float) ref);
  if (fabs((float) ref) < 1e-7f)
  {
    return false;
  }
  float normError = sqrtf((float) error);
  error = normError / normRef;
  return error < epsilon;
}

int main(int argc, char * argv[])
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

  // error if image did not load
  if(!inputBitmap.isLoaded())
  {
    printf("Failed to load input image!");
    return SDK_FAILURE;
  }

  // get width and height of input image
  const int height = inputBitmap.getHeight();
  const int width = inputBitmap.getWidth();
  const int pixelSize = sizeof(uchar4);
  const int imageSize = width * height * pixelSize;
  printf("Image height = %d and width = %d\n", height, width);

  // allocate memory for input image data
  uchar4 *inputImageData  = (uchar4*) malloc (imageSize);
  if (inputImageData == NULL)
    printf("Failed to allocate memory! (inputImageData)");

  // allocate memory for output image data
  uchar4 *outputImageData = (uchar4*) malloc (imageSize);
  if (outputImageData == NULL) 
    printf("Failed to allocate memory! (outputImageData)");

  // initialize the output
  memset(outputImageData, 0, imageSize);

  // get the pointer to pixel data
  uchar4 *pixelData = inputBitmap.getPixels();
  if(pixelData == NULL)
    printf("Failed to read pixel Data!");

  // Copy pixel data into inputImageData
  memcpy(inputImageData, pixelData, imageSize);

  // allocate memory for verification output
  uchar4* verificationOutput = (uchar4*) malloc (imageSize);
  if (verificationOutput == NULL) 
    printf("verificationOutput heap allocation failed!");

  // initialize the output
  memset(verificationOutput, 0, imageSize);

  // Create memory object for input Image
  uchar4 *inputImageBuffer;
  cudaMalloc((void**)&inputImageBuffer, imageSize); 
  cudaMemcpy(inputImageBuffer, inputImageData, imageSize, cudaMemcpyHostToDevice);

  uchar4 *outputImageBuffer;
  cudaMalloc((void**)&outputImageBuffer, imageSize); 
  cudaMemset(outputImageBuffer, 0, imageSize); 

  // Enqueue a kernel run call.
  const int blockSizeX = 16;
  const int blockSizeY = 16;
  dim3 grid (width/blockSizeX, height/blockSizeY);
  dim3 block (blockSizeX, blockSizeY);

  printf("Executing kernel for %d iterations", iterations);
  printf("-------------------------------------------\n");

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
  {
    sobel_filter<<<grid, block>>>(inputImageBuffer, outputImageBuffer, width, height);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / iterations);

  cudaMemcpy(outputImageData, outputImageBuffer, imageSize, cudaMemcpyDeviceToHost);
  cudaFree(inputImageBuffer);
  cudaFree(outputImageBuffer);

  // reference implementation
  reference (verificationOutput, inputImageData, width, height, pixelSize);

  float *outputDevice = (float*) malloc (imageSize * sizeof(float));
  if (outputDevice == NULL)
    printf("Failed to allocate host memory! (outputDevice)");

  float *outputReference = (float*) malloc (imageSize * sizeof(float));

  if (outputReference == NULL)
    printf("Failed to allocate host memory!" "(outputReference)");

  // copy uchar data to float array
  for(int i = 0; i < width * height; i++)
  {
    outputDevice[i * 4 + 0] = outputImageData[i].x;
    outputDevice[i * 4 + 1] = outputImageData[i].y;
    outputDevice[i * 4 + 2] = outputImageData[i].z;
    outputDevice[i * 4 + 3] = outputImageData[i].w;

    outputReference[i * 4 + 0] = verificationOutput[i].x;
    outputReference[i * 4 + 1] = verificationOutput[i].y;
    outputReference[i * 4 + 2] = verificationOutput[i].z;
    outputReference[i * 4 + 3] = verificationOutput[i].w;
  }

  // compare the results and see if they match
  if(compare(outputReference, outputDevice, imageSize))
    printf("PASS\n");
  else
    printf("FAIL\n");
  if (!(compare(outputReference, outputDevice, imageSize))) exit(1);

  free(outputDevice);
  free(outputReference);
  free(verificationOutput);
  free(inputImageData);
  free(outputImageData);
  return SDK_SUCCESS;
}
