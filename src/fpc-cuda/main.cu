#include <stdio.h>      /* defines printf for tests */
#include <stdlib.h> 
#include <chrono>
#include <cuda.h>

typedef unsigned long ulong;

ulong * convertBuffer2Array (char *cbuffer, unsigned size, unsigned step)
{
  unsigned i,j; 
  ulong * values = NULL;
  posix_memalign((void**)&values, 1024, sizeof(ulong)*size/step);
  for (i = 0; i < size / step; i++) {
    values[i] = 0;    // Initialize all elements to zero.
  }
  for (i = 0; i < size; i += step ){
    for (j = 0; j < step; j++){
      values[i / step] += (ulong)((unsigned char)cbuffer[i + j]) << (8*j);
    }
  }
  return values;
}

__host__ __device__
unsigned my_abs ( int x )
{
  unsigned t = x >> 31;
  return (x ^ t) - t;
}

unsigned FPCCompress(ulong *values, unsigned size )
{
  unsigned compressable = 0;
  unsigned i;
  for (i = 0; i < size; i++) {
    // 000
    if(values[i] == 0){
      compressable += 1;
      continue;
    }
    // 001 010
    if(my_abs((int)(values[i])) <= 0xFF){
      compressable += 1;
      continue;
    }
    // 011
    if(my_abs((int)(values[i])) <= 0xFFFF){
      compressable += 2;
      continue;
    }
    //100  
    if(((values[i]) & 0xFFFF) == 0 ){
      compressable += 2;
      continue;
    }
    //101
    if( my_abs((int)((values[i]) & 0xFFFF)) <= 0xFF
        && my_abs((int)((values[i] >> 16) & 0xFFFF)) <= 0xFF){
      compressable += 2;
      continue;
    }
    //110
    unsigned byte0 = (values[i]) & 0xFF;
    unsigned byte1 = (values[i] >> 8) & 0xFF;
    unsigned byte2 = (values[i] >> 16) & 0xFF;
    unsigned byte3 = (values[i] >> 24) & 0xFF;
    if(byte0 == byte1 && byte0 == byte2 && byte0 == byte3){
      compressable += 1;
      continue;
    }
    //111
    compressable += 4;
  }
  return compressable;
}

__device__
unsigned f1(ulong value, bool* mask) {
  if (value == 0) {
    *mask = 1;
  } 
  return 1;
}

__device__
unsigned f2(ulong value, bool* mask) {
  if (my_abs((int)(value)) <= 0xFF) *mask = 1;
  return 1;
}

__device__
unsigned f3(ulong value, bool* mask) {
  if (my_abs((int)(value)) <= 0xFFFF) *mask = 1;
  return 2;
}

__device__
unsigned f4(ulong value, bool* mask) {
  if (((value) & 0xFFFF) == 0 ) *mask = 1;
  return 2;
}

__device__
unsigned f5(ulong value, bool* mask) {
  if ((my_abs((int)((value) & 0xFFFF))) <= 0xFF && 
      my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) 
    *mask = 1;
  return 2;
}

__device__
unsigned f6(ulong value, bool* mask) {
  unsigned byte0 = (value) & 0xFF;
  unsigned byte1 = (value >> 8) & 0xFF;
  unsigned byte2 = (value >> 16) & 0xFF;
  unsigned byte3 = (value >> 24) & 0xFF;
  if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) 
    *mask = 1;
  return 1;
}

__device__
unsigned f7(ulong value, bool* mask) {
  *mask = 1;
  return 4;
}

__global__ void
fpc_kernel (const ulong* values, unsigned *cmp_size)
{
  __shared__ unsigned compressable;
  int lid = threadIdx.x;
  int WGS = blockDim.x;
  int gid = blockIdx.x*WGS+lid;

  ulong value = values[gid];
  unsigned inc;

  // 000
  if (value == 0){
    inc = 1;
  }
  // 001 010
  else if ((my_abs((int)(value)) <= 0xFF)) {
    inc = 1;
  }
  // 011
  else if ((my_abs((int)(value)) <= 0xFFFF)) {
    inc = 2;
  }
  //100  
  else if ((((value) & 0xFFFF) == 0 )) {
    inc = 2;
  }
  //101
  else if ((my_abs((int)((value) & 0xFFFF))) <= 0xFF
      && my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF ) {
    inc = 2;
  }
  //110
  else if( (((value) & 0xFF) == ((value >> 8) & 0xFF)) &&
      (((value) & 0xFF) == ((value >> 16) & 0xFF)) &&
      (((value) & 0xFF) == ((value >> 24) & 0xFF)) ) {
    inc = 1;
  } else { 
    inc = 4;
  }

  if (lid == 0) compressable = 0;
  __syncthreads();

  atomicAdd(&compressable, inc);
  __syncthreads();
  if (lid == WGS-1) {
    atomicAdd(cmp_size, compressable);
  }
}

__global__ void
fpc2_kernel (const ulong* values, unsigned *cmp_size)
{
  __shared__ unsigned compressable;
  int lid = threadIdx.x;
  int WGS = blockDim.x;
  int gid = blockIdx.x*WGS+lid;

  unsigned inc;

  bool m1 = 0;
  bool m2 = 0;
  bool m3 = 0;
  bool m4 = 0;
  bool m5 = 0;
  bool m6 = 0;
  bool m7 = 0;

  ulong value = values[gid];
  unsigned inc1 = f1(value, &m1);
  unsigned inc2 = f2(value, &m2);
  unsigned inc3 = f3(value, &m3);
  unsigned inc4 = f4(value, &m4);
  unsigned inc5 = f5(value, &m5);
  unsigned inc6 = f6(value, &m6);
  unsigned inc7 = f7(value, &m7);

  if (m1)
    inc = inc1;
  else if (m2)
    inc = inc2;
  else if (m3)
    inc = inc3;
  else if (m4)
    inc = inc4;
  else if (m5)
    inc = inc5;
  else if (m6)
    inc = inc6;
  else
    inc = inc7;


  if (lid == 0) compressable = 0;
  __syncthreads();

  atomicAdd(&compressable, inc);
  __syncthreads();
  if (lid == WGS-1) {
    atomicAdd(cmp_size, compressable);
  }
}

void fpc (const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs)
{
  *cmp_size_hw = 0;
  ulong* d_values;
  unsigned* d_cmp_size;
  cudaMalloc((void**)&d_values, values_size*sizeof(ulong));
  cudaMemcpy(d_values, values, values_size*sizeof(ulong), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_cmp_size, sizeof(unsigned));
  cudaMemcpy(d_cmp_size, cmp_size_hw, sizeof(unsigned), cudaMemcpyHostToDevice);

  dim3 grids (values_size/wgs);
  dim3 threads (wgs);

  fpc_kernel<<<grids, threads>>>(d_values, d_cmp_size);

  cudaMemcpy(cmp_size_hw, d_cmp_size, sizeof(unsigned), cudaMemcpyDeviceToHost);
  cudaFree(d_values);
  cudaFree(d_cmp_size);
}

void fpc2 (const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs)
{
  *cmp_size_hw = 0;
  ulong* d_values;
  unsigned* d_cmp_size;
  cudaMalloc((void**)&d_values, values_size*sizeof(ulong));
  cudaMemcpy(d_values, values, values_size*sizeof(ulong), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_cmp_size, sizeof(unsigned));
  cudaMemcpy(d_cmp_size, cmp_size_hw, sizeof(unsigned), cudaMemcpyHostToDevice);

  dim3 grids (values_size/wgs);
  dim3 threads (wgs);

  fpc2_kernel<<<grids, threads>>>(d_values, d_cmp_size);

  cudaMemcpy(cmp_size_hw, d_cmp_size, sizeof(unsigned), cudaMemcpyDeviceToHost);
  cudaFree(d_values);
  cudaFree(d_cmp_size);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <work-group size> <repeat>\n", argv[0]);
    return 1;
  }
  const int wgs = atoi(argv[1]); 
  const int repeat = atoi(argv[2]);

  // create the char buffer
  const int step = 4;
  const size_t size = (size_t)wgs * wgs * wgs;
  char* cbuffer = (char*) malloc (size * step);

  srand(2);
  for (size_t i = 0; i < size*step; i++) {
    cbuffer[i] = 0xFF << (rand() % 256);
  }

  ulong *values = convertBuffer2Array (cbuffer, size, step);
  unsigned values_size = size / step;

  // run on the host
  unsigned cmp_size = FPCCompress(values, values_size);

  // run on the device
  unsigned cmp_size_hw; 

  bool ok = true;

  // warmup
  fpc(values, &cmp_size_hw, values_size, wgs);

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++) {
    fpc(values, &cmp_size_hw, values_size, wgs);
    if (cmp_size_hw != cmp_size) {
      printf("fpc failed %u != %u\n", cmp_size_hw, cmp_size);
      ok = false;
      break;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("fpc: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

  // warmup
  fpc2(values, &cmp_size_hw, values_size, wgs);

  start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++) {
    fpc2(values, &cmp_size_hw, values_size, wgs);
    if (cmp_size_hw != cmp_size) {
      printf("fpc2 failed %u != %u\n", cmp_size_hw, cmp_size);
      ok = false;
      break;
    }
  }

  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("fpc2: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(values);
  free(cbuffer);
  return 0;
}
