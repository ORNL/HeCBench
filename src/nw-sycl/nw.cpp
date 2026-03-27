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
#include <iostream>
#include <string>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

// kernel
#define SCORE(i, j) input_itemsets_l[j + i * (BLOCK_SIZE+1)]
#define REF(i, j)   reference_l[j + i * BLOCK_SIZE]
int max3( int a, int b, int c){

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
  exit(1);
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

  const int matrix_size = max_rows * max_cols;
  const size_t matrix_size_bytes = matrix_size * sizeof(int);

  reference = (int *)malloc( matrix_size_bytes );
  input_itemsets = (int *)malloc( matrix_size_bytes );
  output_itemsets = (int *)malloc( matrix_size_bytes );

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


#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int workgroupsize = BLOCK_SIZE;
#ifdef DEBUG
  if (workgroupsize < 0) {
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

  int *d_input_itemsets_acc = sycl::malloc_device<int>(matrix_size, q);
  q.memcpy(d_input_itemsets_acc, input_itemsets, matrix_size_bytes);

  int *d_reference_acc = sycl::malloc_device<int>(matrix_size, q);
  q.memcpy(d_reference_acc, reference,  matrix_size_bytes);

  int warmup = 100;

  std::chrono::steady_clock::time_point start, end;

  for (int i = 0; i < warmup + repeat; i++) {
    if (i == warmup) {
      q.wait();
      start = std::chrono::steady_clock::now();
    }

    #ifdef DEBUG
      printf("Processing upper-left matrix\n");
    #endif

    for(int blk = 1 ; blk <= block_width ; blk++){
      global_work = BLOCK_SIZE * blk; // kernel arg set every iteration
      #ifdef DEBUG
      printf("global size: %d local size: %d\n", global_work, local_work);
      #endif
      q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor <int, 1> input_itemsets_l (sycl::range<1>((BLOCK_SIZE + 1) *(BLOCK_SIZE+1)), cgh);
        sycl::local_accessor <int, 1> reference_l (sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
        cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(global_work), sycl::range<1>(local_work)), [=] (sycl::nd_item<1> item) {
            #include "kernel1.sycl"
        });
      });
    }

    #ifdef DEBUG
    printf("Processing lower-right matrix\n");
    #endif

    for(int blk = block_width - 1 ; blk >= 1 ; blk--){
      global_work = BLOCK_SIZE * blk;
      q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor <int, 1> input_itemsets_l (sycl::range<1>((BLOCK_SIZE + 1) *(BLOCK_SIZE+1)), cgh);
        sycl::local_accessor <int, 1> reference_l (sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
        cgh.parallel_for<class kernel2>(
          sycl::nd_range<1>(sycl::range<1>(global_work), sycl::range<1>(local_work)), [=] (sycl::nd_item<1> item) {
            #include "kernel2.sycl"
        });
      });
    }
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f / repeat);

  q.memcpy(output_itemsets, d_input_itemsets_acc, matrix_size_bytes).wait();

  // verify
  nw_host(input_itemsets, reference, max_cols, penalty);
  int err = memcmp(input_itemsets, output_itemsets, max_cols * max_rows * sizeof(int));
  printf("%s\n", err ? "FAIL" : "PASS");

#ifdef TRACEBACK

#ifdef USE_GPU
  FILE *fpo = fopen("gpu_result.txt","w");
#else
  FILE *fpo = fopen("cpu_result.txt","w");
#endif
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

    //traceback = max3(nw, w, n);
    int new_nw, new_w, new_n;
    new_nw = nw + reference[i * max_cols + j];
    new_w = w - penalty;
    new_n = n - penalty;

    traceback = max3(new_nw, new_w, new_n);
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
  sycl::free(d_input_itemsets_acc, q);
  sycl::free(d_reference_acc, q);
  return 0;
}
