#ifndef _C_UTIL_
#define _C_UTIL_
#include <math.h>
#include <iostream>

//-------------------------------------------------------------------
//--initialize array with maximum limit
//-------------------------------------------------------------------
template<typename datatype>
void fill(datatype *A, const int n, const datatype maxi){
  for (int j = 0; j < n; j++) 
  {
    A[j] = ((datatype) maxi * (rand() / (RAND_MAX + 1.0f)));
  }
}

//--print matrix
template<typename datatype>
void print_matrix(datatype *A, int height, int width){
  for(int i=0; i<height; i++){
    for(int j=0; j<width; j++){
      int idx = i*width + j;
      std::cout << A[idx] << " ";
    }
    std::cout << std::endl;
  }
}

template<typename datatype>
void compare_results(const datatype *cpu_results, const datatype *gpu_results, const int size) {

  char passed = true; 
  for (int i=0; i<size; i++){
    if (cpu_results[i]!=gpu_results[i])
      passed = false; 
  }
  std::cout << (passed ? "PASS" : "FAIL") << std::endl;
}

#endif

