//
// GPU implementation of FIM (Fast Iterative Method) for Eikonal equations
//
// Copyright (c) Won-Ki Jeong (wkjeong@unist.ac.kr)
//
// 2016. 2. 4
//

#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <cstdio>
#include "common_def.h"

#define MEM(index) _mem[index]
#define SOL(i,j,k) _sol[i][j][k]
#define SPD(i,j,k) _spd[i][j][k]

#define __syncthreads() item.barrier(sycl::access::fence_space::local_space)

DOUBLE get_time_eikonal(DOUBLE a, DOUBLE b, DOUBLE c, DOUBLE s);
//
// F : Input speed (positive)
// if F =< 0, skip that pixel (masking out)
//
SYCL_EXTERNAL
void run_solver(
  sycl::nd_item<3> &item,
  DOUBLE *__restrict__ sol_mem,
  const double*__restrict__ spd,
  const bool*__restrict__ mask,
  const DOUBLE *__restrict__ sol_in,
  DOUBLE *__restrict__ sol_out,
  bool *__restrict__ con,
  const uint*__restrict__ list,
  int xdim, int ydim, int zdim,
  int nIter, uint nActiveBlock);
//
// run_reduction
//
// con is pixelwise convergence. Do reduction on active tiles and write tile-wise
// convergence to listVol. The implementation assumes that the block size is 4x4x4.
//
SYCL_EXTERNAL
void run_reduction(
  sycl::nd_item<3> &item,
  bool *__restrict__ conv,
  const bool *__restrict__ con,
  bool *__restrict__ listVol,
  const uint *__restrict__ list,
  uint nActiveBlock);
//
// if block is active block, copy values
// if block is neighbor, run solver once
//
SYCL_EXTERNAL
void run_check_neighbor(
  sycl::nd_item<3> &item,
  DOUBLE *__restrict__ sol_mem,
  const double*__restrict__ spd,
  const bool*__restrict__ mask,
  const DOUBLE *__restrict__ sol_in,
  DOUBLE *__restrict__ sol_out,
  bool *__restrict__ con,
  const uint*__restrict__ list,
  int xdim, int ydim, int zdim,
  uint nActiveBlock, uint nTotalBlock);

#endif


