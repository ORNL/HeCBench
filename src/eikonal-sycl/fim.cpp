//
// GPU implementation of FIM (Fast Iterative Method) for Eikonal equations
//
// Copyright (c) Won-Ki Jeong (wkjeong@unist.ac.kr)
//
// 2016. 2. 4
//

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include "kernel.h"
#include "fim.h"

void runEikonalSolverSimple(sycl::queue &q, GPUMEMSTRUCT &cmem)
{
  int xdim, ydim, zdim;
  xdim = cmem.xdim;
  ydim = cmem.ydim;
  zdim = cmem.zdim;

  // create volumes
  uint volSize = cmem.volsize;
  uint blockNum = cmem.blknum;

  printf("# of total voxels : %d\n", volSize);
  printf("# of total blocks : %d\n", blockNum);

  // h_ : host memory, d_ : device memory

  // copy speed table to constant variable
  int nIter = cmem.nIter;
  uint nActiveBlock = cmem.nActiveBlock; // active list

  double *d_spd = cmem.d_spd;
  DOUBLE *d_sol = cmem.d_sol;
  DOUBLE *t_sol = cmem.t_sol;

  uint *d_list = cmem.d_list;
  bool *d_listVol = cmem.d_listVol;

  bool *d_con = cmem.d_con;
  bool *d_mask = cmem.d_mask;

  // copy so that original value should not be modified
  uint *h_list = (uint*) malloc(blockNum*sizeof(uint));
  bool *h_listed = (bool*) malloc(blockNum*sizeof(bool));
  bool *h_listVol = (bool*) malloc(blockNum*sizeof(bool));

  // initialization
  memcpy(h_list, cmem.h_list, blockNum*sizeof(uint));
  memcpy(h_listed, cmem.h_listed, blockNum*sizeof(bool));
  memcpy(h_listVol, cmem.h_listVol, blockNum*sizeof(bool));

  q.memcpy(cmem.d_list, cmem.h_list, nActiveBlock*sizeof(uint));
  q.memcpy(cmem.d_listVol, cmem.h_listVol, blockNum*sizeof(bool));
  q.memcpy(cmem.d_sol, cmem.h_sol, volSize*sizeof(DOUBLE));
  q.memcpy(cmem.t_sol, cmem.h_sol, volSize*sizeof(DOUBLE));
  q.memset(cmem.d_con, 1, volSize*sizeof(bool));

  // set dimension of block and entire grid size
  sycl::range<3> grids (1, 1, nActiveBlock);
  sycl::range<3> blocks (BLOCK_LENGTH, BLOCK_LENGTH, BLOCK_LENGTH);

  int nTotalIter = 0;
  //uint sharedmemsize = sizeof(float)*BLOCK_LENGTH*BLOCK_LENGTH*(3*BLOCK_LENGTH + 2);

  std::vector<int> sourceList;
  sourceList.push_back((zdim/2)*ydim*xdim + (ydim/2)*xdim + (xdim/2));

#ifdef TIMER
  // initialize & start timer
  StopWatchInterface *timer_total, *timer_solver, *timer_reduction, *timer_list, *timer_list2, *timer_coarse;
  timer_total = timer_solver = timer_reduction = timer_list = timer_list2 = timer_coarse = NULL;

  sdkCreateTimer(&timer_total);
  sdkCreateTimer(&timer_solver);
  sdkCreateTimer(&timer_reduction);
  sdkCreateTimer(&timer_list);
  sdkCreateTimer(&timer_list2);
  sdkCreateTimer(&timer_coarse);
  sdkStartTimer(&timer_total);

  q.wait();
#endif

  uint nTotalBlockProcessed = 0;

  // start solver
  while(nActiveBlock > 0)
  {
    assert(nActiveBlock < 4294967295);

    nTotalBlockProcessed += nActiveBlock;

    nTotalIter++;

    //
    // solve current blocks in the active lists
    //

#ifdef DEBUG
      printf("# of active tiles : %u\n", nActiveBlock);
#endif

    //////////////////////////////////////////////////////////////////
    // 1. run solver on current active tiles

#ifdef TIMER
    sdkStartTimer(&timer_solver);
#endif

    //dimGrid.y = (unsigned int)floor(((double)nActiveBlock-1)/65535)+1;
    //dimGrid.x = (unsigned int)ceil ((double)nActiveBlock/(double)dimGrid.y);

    grids[1] = (unsigned int)floor(((double)nActiveBlock-1)/65535)+1;
    grids[2] = (unsigned int)ceil ((double)nActiveBlock/(double)grids[1]);

#ifdef DEBUG
      printf("Grid size : %d x %d\n", grids[2], grids[1]);
#endif

    q.memcpy(d_list, h_list, nActiveBlock*sizeof(uint));

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<DOUBLE, 1> sol_local(
        sycl::range<1>((BLOCK_LENGTH+2)*(BLOCK_LENGTH+2)*(BLOCK_LENGTH+2)), cgh);
      cgh.parallel_for<class solve>(
        sycl::nd_range<3>(grids*blocks, blocks), [=] (sycl::nd_item<3> item) {
        run_solver(item, sol_local.get_multi_ptr<sycl::access::decorated::no>().get(),
                   d_spd, d_mask, d_sol, t_sol, d_con,
                   d_list, xdim, ydim, zdim, nIter, nActiveBlock);
      });
    });

#ifdef TIMER
    q.wait();
    sdkStopTimer(&timer_solver);
#endif

    //////////////////////////////////////////////////////////////////
    // 2. reduction (only active tiles)

#ifdef TIMER
    sdkStartTimer(&timer_reduction);
#endif
    
    sycl::range<3> lws_reduce (BLOCK_LENGTH/2, BLOCK_LENGTH, BLOCK_LENGTH);

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<bool, 1> conv_local(
        sycl::range<1>(BLOCK_LENGTH*BLOCK_LENGTH*BLOCK_LENGTH), cgh);
      cgh.parallel_for<class reduce>(
        sycl::nd_range<3>(grids*lws_reduce, lws_reduce), [=] (sycl::nd_item<3> item) {
        run_reduction(item, conv_local.get_multi_ptr<sycl::access::decorated::no>().get(),
                      d_con, d_listVol, d_list, nActiveBlock);
      });
    });

#ifdef TIMER
    q.wait();
    sdkStopTimer(&timer_reduction);
#endif

    //////////////////////////////////////////////////////////////////
    // 3. check neighbor tiles of converged tile
    // Add any active block of neighbor of converged block is inserted
    // to the list

    // read-back active list volume
#ifdef TIMER
    sdkStartTimer(&timer_list);
#endif

    q.memcpy(h_listVol, d_listVol, blockNum*sizeof(bool)).wait();

    uint nOldActiveBlock = nActiveBlock;
    uint nBlkX = xdim/BLOCK_LENGTH;
    uint nBlkY = ydim/BLOCK_LENGTH;

    for(uint i=0; i<nOldActiveBlock; i++)
    {
      // check 6-neighbor of current active tile
      uint currBlkIdx = h_list[i];

      if(!h_listVol[currBlkIdx]) // not active : converged
      {
        uint nb[6];
        nb[0] = (currBlkIdx < nBlkX*nBlkY) ? currBlkIdx : (currBlkIdx - nBlkX*nBlkY);  //tp
        nb[1] = ((currBlkIdx + nBlkX*nBlkY) >= blockNum) ? currBlkIdx : (currBlkIdx + nBlkX*nBlkY); //bt
        nb[2] = (currBlkIdx < nBlkX) ? currBlkIdx : (currBlkIdx - nBlkX); //up
        nb[3] = ((currBlkIdx + nBlkX) >= blockNum) ? currBlkIdx : (currBlkIdx + nBlkX); //dn
        nb[4] = (currBlkIdx%nBlkX == 0) ? currBlkIdx : currBlkIdx-1; //lf
        nb[5] = ((currBlkIdx+1)%nBlkX == 0) ? currBlkIdx : currBlkIdx+1; //rt

        for(int nbIdx = 0; nbIdx < 6; nbIdx++)
        {
          uint currIdx = nb[nbIdx];

          //  assert(currIdx < volSize);

          if(!h_listed[currIdx])
          {
            h_listed[currIdx] = true;
            h_list[nActiveBlock++] = currIdx;
          }
        }
      }
    }

#ifdef TIMER
    q.wait();
    sdkStopTimer(&timer_list);
#endif

    //////////////////////////////////////////////////////////////////
    // 4. run solver only once for neighbor blocks of converged block
    // current active list contains active blocks and neighbor blocks of
    // any converged blocks.
    //

#ifdef TIMER
    sdkStartTimer(&timer_solver);
#endif

    // update grid dimension because nActiveBlock is changed
    grids[1] = (unsigned int)floor(((double)nActiveBlock-1)/65535)+1;
    grids[2] = (unsigned int)ceil ((double)nActiveBlock/(double)grids[1]);

#ifdef DEBUG
      printf("Grid size : %d x %d\n", grids[2], grids[1]);
#endif

    q.memcpy(d_list, h_list, nActiveBlock*sizeof(uint)).wait();

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<DOUBLE, 1> sol_local(
        sycl::range<1>((BLOCK_LENGTH+2)*(BLOCK_LENGTH+2)*(BLOCK_LENGTH+2)), cgh);
      cgh.parallel_for<class check_neighbor>(
        sycl::nd_range<3>(grids*blocks, blocks), [=] (sycl::nd_item<3> item) {
        run_check_neighbor(item, sol_local.get_multi_ptr<sycl::access::decorated::no>().get(),
                           d_spd, d_mask, t_sol, d_sol, d_con, d_list,
                           xdim, ydim, zdim, nOldActiveBlock, nActiveBlock);
      });
    });

#ifdef TIMER
    q.wait();
    sdkStopTimer(&timer_solver);
#endif

    //////////////////////////////////////////////////////////////////
    // 5. reduction

#ifdef TIMER
    sdkStartTimer(&timer_reduction);
#endif

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<bool, 1> conv_local(
        sycl::range<1>(BLOCK_LENGTH*BLOCK_LENGTH*BLOCK_LENGTH), cgh);
      cgh.parallel_for<class reduce2>(
        sycl::nd_range<3>(grids*lws_reduce, lws_reduce), [=] (sycl::nd_item<3> item) {
        run_reduction(item, conv_local.get_multi_ptr<sycl::access::decorated::no>().get(),
                      d_con, d_listVol, d_list, nActiveBlock);
      });
    });

#ifdef TIMER
    q.wait();
    sdkStopTimer(&timer_reduction);
#endif

    //////////////////////////////////////////////////////////////////
    // 6. update active list
    // read back active volume from the device and add
    // active block to active list on the host memory

#ifdef TIMER
    sdkStartTimer(&timer_list2);
#endif

    nActiveBlock = 0;
    q.memcpy(h_listVol, d_listVol, blockNum*sizeof(bool)).wait();

    for(uint i=0; i<blockNum; i++)
    {
      if(h_listVol[i]) // true : active block (not converged)
      {
        h_listed[i] = true;
        h_list[nActiveBlock++] = i;
        //printf("Block %d added\n", i);
      }
      else h_listed[i] = false;
    }

#ifdef TIMER
    q.wait();
    sdkStopTimer(&timer_list2);
#endif

#ifdef DEBUG
      printf("Iteration : %d\n", nTotalIter);
#endif
  }

#ifdef TIMER
  sdkStopTimer(&timer_total);
#endif

  printf("Eikonal solver converged after %d iterations\n", nTotalIter);

#ifdef TIMER
  printf("Total Running Time: %f (sec)\n", sdkGetTimerValue(&timer_total) / 1000);
  printf("Time for solver : %f (sec)\n", sdkGetTimerValue(&timer_solver) / 1000);
  printf("Time for reduction : %f (sec)\n", sdkGetTimerValue(&timer_reduction) / 1000);
  printf("Time for list update-1 (CPU) : %f (sec)\n", sdkGetTimerValue(&timer_list) / 1000);
  printf("Time for list update-2 (CPU) : %f (sec)\n", sdkGetTimerValue(&timer_list2) / 1000);
#endif

  printf("Total # of blocks processed : %d\n", nTotalBlockProcessed);


  // delete dynamically allocated host memory
  free(h_list);
  free(h_listed);
  free(h_listVol);
}
