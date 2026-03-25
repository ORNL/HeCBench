#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif

#define LIMIT -999

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include "reference.h"

// kernel
#define SCORE(i, j) input_itemsets_l[j + i * (BLOCK_SIZE+1)]
#define REF(i, j)   reference_l[j + i * BLOCK_SIZE]

__device__ __host__
int maximum( int a, int b, int c){

  int k;
  if( a <= b )
    k = b;
  else
    k = a;
  if( k <=c )
    return(c);
  else
    return(k);
}

//global variables

int blosum62[24][24] = {
  { 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
  {-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
  {-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
  {-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
  { 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
  {-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
  {-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
  { 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
  {-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
  {-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
  {-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
  {-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
  {-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
  {-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
  {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
  { 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
  { 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
  {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
  {-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
  { 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
  {-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
  {-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
  { 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
  {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

// local variables

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <repeat>\n", argv[0]);
  fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
  fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
  fprintf(stderr, "\t<repeat> - the number of kernel executions\n");
  fprintf(stderr, "\t<file> - filename\n");
  exit(1);
}

__global__ void
kernel1 (int*__restrict__ d_input_itemsets,
         const int*__restrict__ d_reference,
         const int offset_r,
         const int offset_c,
         const int max_cols,
         const int blk,
         const int penalty)
{
  __shared__ int input_itemsets_l [(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)];
  __shared__ int reference_l [BLOCK_SIZE*BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  // Base elements
  int base = offset_r * max_cols + offset_c;

  int b_index_x = bx;
  int b_index_y = blk - 1 - bx;

  int index   =   base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( max_cols + 1 );
  int index_n   = base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  int index_w   = base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( max_cols );
  int index_nw =  base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

  if (tx == 0) SCORE(tx, 0) = d_input_itemsets[index_nw + tx];

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)  {
    REF(ty, tx) =  d_reference[index + max_cols * ty];
  }

  SCORE((tx + 1), 0) = d_input_itemsets[index_w + max_cols * tx];

  SCORE(0, (tx + 1)) = d_input_itemsets[index_n];

  __syncthreads();

  for( int m = 0 ; m < BLOCK_SIZE ; m++){
     if ( tx <= m ){
        int t_index_x =  tx + 1;
        int t_index_y =  m - tx + 1;

        SCORE(t_index_y, t_index_x) = maximum( SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
              SCORE((t_index_y),   (t_index_x-1)) - (penalty),
              SCORE((t_index_y-1), (t_index_x))   - (penalty));
     }
     __syncthreads();
  }

  for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){

     if ( tx <= m){
        int t_index_x =  tx + BLOCK_SIZE - m ;
        int t_index_y =  BLOCK_SIZE - tx;

        SCORE(t_index_y, t_index_x) = maximum(  SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
              SCORE((t_index_y),   (t_index_x-1)) - (penalty),
              SCORE((t_index_y-1), (t_index_x))   - (penalty));
     }
     __syncthreads();
  }

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++) {
     d_input_itemsets[index + max_cols * ty] = SCORE((ty+1), (tx+1));
  }

}

__global__ void
kernel2 (int*__restrict__ d_input_itemsets,
         const int*__restrict__ d_reference,
         const int block_width,
         const int offset_r,
         const int offset_c,
         const int max_cols,
         const int blk,
         const int penalty)
{
   __shared__ int input_itemsets_l [(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)];
   __shared__ int reference_l [BLOCK_SIZE*BLOCK_SIZE];
   int bx = blockIdx.x;
   int tx = threadIdx.x;

   // Base elements
   int base = offset_r * max_cols + offset_c;
   int b_index_x = bx + block_width - blk  ;
   int b_index_y = block_width - bx -1;

   int index   =   base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( max_cols + 1 );
   int index_n   = base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
   int index_w   = base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( max_cols );
   int index_nw =  base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

   if (tx == 0)
      SCORE(tx, 0) = d_input_itemsets[index_nw];

   for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
      REF(ty, tx) =  d_reference[index + max_cols * ty];

   SCORE((tx + 1), 0) = d_input_itemsets[index_w + max_cols * tx];

   SCORE(0, (tx + 1)) = d_input_itemsets[index_n];

   __syncthreads();

   for( int m = 0 ; m < BLOCK_SIZE ; m++){
      if ( tx <= m ){

         int t_index_x =  tx + 1;
         int t_index_y =  m - tx + 1;

         SCORE(t_index_y, t_index_x) = maximum(  SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
               SCORE((t_index_y),   (t_index_x-1)) - (penalty),
               SCORE((t_index_y-1), (t_index_x))   - (penalty));
      }
      __syncthreads();
   }

   for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){

      if ( tx <= m){

         int t_index_x =  tx + BLOCK_SIZE - m ;
         int t_index_y =  BLOCK_SIZE - tx;

         SCORE(t_index_y, t_index_x) = maximum( SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
               SCORE((t_index_y),   (t_index_x-1)) - (penalty),
               SCORE((t_index_y-1), (t_index_x))   - (penalty));

      }
      __syncthreads();
   }

   for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
      d_input_itemsets[index + ty * max_cols] = SCORE((ty+1), (tx+1));
}

int main(int argc, char **argv){

  printf("WG size of kernel = %d \n", BLOCK_SIZE);

  int max_rows_t, max_cols_t, penalty_t;
  int repeat;
  // the lengths of the two sequences should be able to divided by 16.
  // And at current stage  max_rows needs to equal max_cols
  if (argc == 4)
  {
    max_rows_t = atoi(argv[1]);
    max_cols_t = atoi(argv[1]);
    penalty_t = atoi(argv[2]);
    repeat = atoi(argv[3]);
  }
  else{
    usage(argc, argv);
  }

  if(atoi(argv[1])%16!=0){
    fprintf(stderr,"The dimension values must be a multiple of 16\n");
    exit(1);
  }

  // make constant variable to avoid kernel argument set at every loop iteration
  const int max_rows = max_rows_t + 1;
  const int max_cols = max_cols_t + 1;
  const int penalty = penalty_t;

  int *reference;
  int *input_itemsets;
  int *output_itemsets;

  reference = (int *)malloc( max_rows * max_cols * sizeof(int) );
  input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
  output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );

  srand(7);

  //initialization
  for (int i = 0 ; i < max_cols; i++){
    for (int j = 0 ; j < max_rows; j++){
      input_itemsets[i*max_cols+j] = 0;
    }
  }

  for( int i=1; i< max_rows ; i++){    //initialize the cols
    input_itemsets[i*max_cols] = rand() % 10 + 1;
  }

  for( int j=1; j< max_cols ; j++){    //initialize the rows
    input_itemsets[j] = rand() % 10 + 1;
  }

  for (int i = 1 ; i < max_cols; i++){
    for (int j = 1 ; j < max_rows; j++){
      reference[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
    }
  }

  for( int i = 1; i< max_rows ; i++)
    input_itemsets[i*max_cols] = -i * penalty;
  for( int j = 1; j< max_cols ; j++)
    input_itemsets[j] = -j * penalty;

  int workgroupsize = BLOCK_SIZE;
#ifdef DEBUG
  if(workgroupsize < 0){
     printf("ERROR: invalid or missing <num_work_items>[/<work_group_size>]\n");
     return -1;
  }
#endif
  // set global and local workitems
  const size_t local_work = (size_t)workgroupsize;
  size_t global_work;

  const int worksize = max_cols - 1;
#ifdef DEBUG
  printf("worksize = %d\n", worksize);
#endif
  //these two parameters are for extension use, don't worry about it.
  const int offset_r = 0;
  const int offset_c = 0;
  const int block_width = worksize/BLOCK_SIZE ;
  printf("block width = %d\n", block_width);

  int *d_input_itemsets;
  int *d_reference;
  cudaMalloc((void**)&d_input_itemsets, max_cols * max_rows * sizeof(int));
  cudaMalloc((void**)&d_reference, max_cols * max_rows * sizeof(int));

  cudaMemcpy(d_input_itemsets, input_itemsets, max_cols * max_rows * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_reference, reference, max_cols * max_rows * sizeof(int), cudaMemcpyHostToDevice);

  int warmup = 100;

  std::chrono::steady_clock::time_point start, end;

  for (int i = 0; i < warmup + repeat; i++) {
    if (i == warmup) {
      cudaDeviceSynchronize();
      start = std::chrono::steady_clock::now();
    }

    #ifdef DEBUG
    printf("Processing upper-left matrix\n");
    #endif
    for( int blk = 1 ; blk <= block_width ; blk++){
      global_work = blk;
      kernel1<<<global_work, local_work>>>(d_input_itemsets, d_reference, offset_r, offset_c, max_cols, blk, penalty);
    }

    #ifdef DEBUG
    printf("Processing lower-right matrix\n");
    #endif
    for( int blk = block_width - 1 ; blk >= 1 ; blk--){
      global_work = blk;
      kernel2<<<global_work, local_work>>>(d_input_itemsets, d_reference, block_width, offset_r, offset_c, max_cols, blk, penalty);
    }
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f / repeat);

  cudaMemcpy(output_itemsets, d_input_itemsets, max_cols * max_rows * sizeof(int), cudaMemcpyDeviceToHost);

  // verify
  nw_host(input_itemsets, reference, max_cols, penalty);
  int err = memcmp(input_itemsets, output_itemsets, max_cols * max_rows * sizeof(int));
  printf("%s\n", err ? "FAIL" : "PASS");

#ifdef TRACEBACK

  FILE *fpo = fopen("result.txt","w");
  fprintf(fpo, "print traceback value:\n");

  for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
    int nw, n, w, traceback;
    if ( i == max_rows - 2 && j == max_rows - 2 )
      fprintf(fpo, "%d ", output_itemsets[ i * max_cols + j]); //print the first element
    if ( i == 0 && j == 0 )
      break;
    if ( i > 0 && j > 0 ){
      nw = output_itemsets[(i - 1) * max_cols + j - 1];
      w  = output_itemsets[ i * max_cols + j - 1 ];
      n  = output_itemsets[(i - 1) * max_cols + j];
    }
    else if ( i == 0 ){
      nw = n = LIMIT;
      w  = output_itemsets[ i * max_cols + j - 1 ];
    }
    else if ( j == 0 ){
      nw = w = LIMIT;
      n  = output_itemsets[(i - 1) * max_cols + j];
    }
    else{
    }

    int new_nw, new_w, new_n;
    new_nw = nw + reference[i * max_cols + j];
    new_w = w - penalty;
    new_n = n - penalty;

    traceback = maximum(new_nw, new_w, new_n);
    if(traceback == new_nw)
      traceback = nw;
    if(traceback == new_w)
      traceback = w;
    if(traceback == new_n)
      traceback = n;

    fprintf(fpo, "%d ", traceback);

    if(traceback == nw )
    {i--; j--; continue;}

    else if(traceback == w )
    {j--; continue;}

    else if(traceback == n )
    {i--; continue;}

    else
      ;
  }

  fclose(fpo);

#endif

  //printf("Computation Done\n");

  free(reference);
  free(input_itemsets);
  free(output_itemsets);
  cudaFree(d_input_itemsets);
  cudaFree(d_reference);
  return 0;
}

