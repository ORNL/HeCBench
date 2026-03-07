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
#include "RadixSort.h"
#include "Scan.h"
#include <cstdlib>

void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits);
bool verifySortUint(unsigned int *keysSorted, 
    unsigned int *valuesSorted, 
    unsigned int *keysUnsorted, 
    unsigned int len);

int main(int argc, const char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int numIterations = atoi(argv[1]);

  const unsigned int numElements = 128*128*128*2;
  const int keybits = 32;  // bit size of uint 
  const int batchSize = 1; // only support a batch size of 1
  const unsigned int numBlocks = ((numElements % (CTA_SIZE * 4)) == 0) ? 
    (numElements / (CTA_SIZE * 4)) : (numElements / (CTA_SIZE * 4) + 1);

  // Check power-of-two factorization before the scan operations start
  unsigned int arrayLength = numElements/2/CTA_SIZE*16;
  unsigned int log2L;
  unsigned int factorizationRemainder = factorRadix2(log2L, arrayLength);
  assert(factorizationRemainder == 1);

  //Check supported size range
  assert((arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE));
  assert(arrayLength > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE);

  //Check total batch size limit
  assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

  // Alloc and init some data on the host, then alloc and init GPU buffer  
  unsigned int* h_keys       = (unsigned int*)malloc(numElements * sizeof(unsigned int));
  unsigned int* h_keysSorted = (unsigned int*)malloc(numElements * sizeof(unsigned int));
  makeRandomUintVector(h_keys, numElements, keybits);

#ifdef DEBUG
  printf("#elements: %u  #blocks: %u\n", numElements, numBlocks);
  for (int i = 0; i < numElements; i++) printf("init key %d: %x\n", i, h_keys[i]);
#endif

  unsigned int* d_keys;
  hipMalloc((void**)&d_keys, numElements*sizeof(unsigned int));
  hipMemcpy(d_keys, h_keys, numElements*sizeof(unsigned int), hipMemcpyHostToDevice);
  
  unsigned int* d_tempKeys;
  hipMalloc((void**)&d_tempKeys, numElements*sizeof(unsigned int));

  unsigned int* d_counters;
  hipMalloc((void**)&d_counters, WARP_SIZE*numBlocks*sizeof(unsigned int));

  unsigned int* d_countersSum;
  hipMalloc((void**)&d_countersSum, WARP_SIZE*numBlocks*sizeof(unsigned int));

  unsigned int* d_blockOffsets;
  hipMalloc((void**)&d_blockOffsets, WARP_SIZE*numBlocks*sizeof(unsigned int));

  // Allocate the buffer once though it is internally used by the scan operations
  unsigned int* d_buffer;  
  hipMalloc((void**)&d_buffer, sizeof(unsigned int) * 
             (arrayLength / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE));

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numIterations; i++)
  {
    radixSortKeys(d_keys, d_tempKeys, d_counters, d_blockOffsets, d_countersSum, 
                  d_buffer, numElements, keybits, batchSize);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of radixsort: %f (s)\n", (time * 1e-9f) / numIterations);

  // copy sorted keys to CPU 
  hipMemcpy(h_keysSorted, d_keys, numElements*sizeof(unsigned int), hipMemcpyDeviceToHost);

#ifdef DEBUG
  for (int i = 0; i < numElements; i++) printf("sorted key %d: %x\n", i, h_keysSorted[i]);
#endif

  // Check results
  bool passed = true;
  passed &= verifySortUint(h_keysSorted, NULL, h_keys, numElements);

  free(h_keys);
  free(h_keysSorted);
  hipFree(d_keys);
  hipFree(d_tempKeys);
  hipFree(d_counters);
  hipFree(d_countersSum);
  hipFree(d_blockOffsets);
  hipFree(d_buffer);

  // finish
  if (passed)
    printf("PASS\n");
  else 
    printf("FAIL\n");
  if (!passed) exit(1);

  return 0;
}

void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits)
{
  // Fill up with some random data
  int keyshiftmask = 0;
  if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
  int keymask = 0xffff;
  if (keybits < 16) keymask = (1 << keybits) - 1;

  srand(95123);
  for(unsigned int i=0; i < numElements; ++i)   
  { 
    a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
  }
}

// assumes the values were initially indices into the array, for simplicity of 
// checking correct order of values
bool verifySortUint(unsigned int *keysSorted, 
    unsigned int *valuesSorted, 
    unsigned int *keysUnsorted, 
    unsigned int len)
{
  bool passed = true;
  for(unsigned int i=0; i<len-1; ++i)
  {
    if( (keysSorted[i])>(keysSorted[i+1]) )
    {
      printf("Unordered key[%d]: %d > key[%d]: %d\n", i, keysSorted[i], i+1, keysSorted[i+1]);
      passed = false;
      break;
    }
  }

  if (valuesSorted)
  {
    for(unsigned int i=0; i<len; ++i)
    {
      if( keysUnsorted[valuesSorted[i]] != keysSorted[i] )
      {
        printf("Incorrectly sorted value[%u] (%u): %u != %u\n", 
            i, valuesSorted[i], keysUnsorted[valuesSorted[i]], keysSorted[i]);
        passed = false;
        break;
      }
    }
  }

  return passed;
}
