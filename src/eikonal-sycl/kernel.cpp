//
// GPU implementation of FIM (Fast Iterative Method) for Eikonal equations
//
// Copyright (c) Won-Ki Jeong (wkjeong@unist.ac.kr)
//
// 2016. 2. 4
//

#include "kernel.h"

DOUBLE get_time_eikonal(DOUBLE a, DOUBLE b, DOUBLE c, DOUBLE s)
{
  DOUBLE ret, tmp;

  // a > b > c
  if(a < b) { tmp = a; a = b; b = tmp; }
  if(b < c) { tmp = b; b = c; c = tmp; }
  if(a < b) { tmp = a; a = b; b = tmp; }

  ret = INF;

  if(c < INF)
  {
    ret = c + s;

    if(ret > b) 
    {  
      tmp = ((b+c) + sycl::sqrt(2.0f*s*s-(b-c)*(b-c)))*0.5f;

      if(tmp > b) ret = tmp; 

      if(ret > a)  {      
        tmp = (a+b+c)/3.0f + sycl::sqrt(2.0f*(a*(b-a)+b*(c-b)+c*(a-c))+3.0f*s*s)/3.0f; 

        if(tmp > a) ret = tmp;
      }
    }
  }

  return ret;
}

SYCL_EXTERNAL
void run_solver(
  sycl::nd_item<3> &item,
  DOUBLE *__restrict sol_mem,
  const double*__restrict spd,
  const bool*__restrict mask,
  const DOUBLE *__restrict sol_in,
  DOUBLE *__restrict sol_out,
  bool *__restrict con,
  const uint*__restrict list,
  int xdim, int ydim, int zdim,
  int nIter, uint nActiveBlock)
{
  uint list_idx = item.get_group(1)*item.get_group_range(2) + item.get_group(2);

  if(list_idx < nActiveBlock)
  {
    // retrieve actual block index from the active list
    uint block_idx = list[list_idx];

    double F;
    bool isValid;
    uint blocksize = BLOCK_LENGTH*BLOCK_LENGTH*BLOCK_LENGTH;
    uint base_addr = block_idx*blocksize;

    uint xgridlength = xdim/BLOCK_LENGTH;
    uint ygridlength = ydim/BLOCK_LENGTH;
    uint zgridlength = zdim/BLOCK_LENGTH;

    // compute block index
    uint bx = block_idx%xgridlength;
    uint tmpIdx = (block_idx - bx)/xgridlength;
    uint by = tmpIdx%ygridlength;
    uint bz = (tmpIdx-by)/ygridlength;

    uint tx = item.get_local_id(2);
    uint ty = item.get_local_id(1);
    uint tz = item.get_local_id(0);
    uint tIdx = tz*BLOCK_LENGTH*BLOCK_LENGTH + ty*BLOCK_LENGTH + tx;

    DOUBLE (*_sol)[BLOCK_LENGTH+2][BLOCK_LENGTH+2] =
      reinterpret_cast<DOUBLE (*)[BLOCK_LENGTH+2][BLOCK_LENGTH+2]>(sol_mem);

    // copy global to shared memory
    //dim3 idx(tx+1,ty+1,tz+1);
    sycl::range<3> idx (tz+1,ty+1,tx+1);

    SOL(idx[2],idx[1],idx[0]) = sol_in[base_addr + tIdx];
    F = spd[base_addr + tIdx];
    if(F > 0) F = 1.0/F; // F = 1/f
    isValid = mask[base_addr + tIdx];

    uint new_base_addr, new_tIdx;

    // 1-neighborhood values
    if(tx == 0) 
    {
      if(bx == 0) // end of the grid
      {  
        new_tIdx = tIdx;
        new_base_addr = base_addr;
      }
      else
      {
        new_tIdx = tIdx + BLOCK_LENGTH-1;
        new_base_addr = (block_idx - 1)*blocksize;  
      }

      SOL(tx,idx[1],idx[0]) = sol_in[new_base_addr + new_tIdx];  
    }

    if(tx == BLOCK_LENGTH-1)
    {
      if(bx == xgridlength-1) // end of the grid
      {
        new_tIdx = tIdx;
        new_base_addr = base_addr;
      }
      else
      {
        new_tIdx = tIdx - (BLOCK_LENGTH-1);
        new_base_addr = (block_idx + 1)*blocksize;  
      }
      SOL(tx+2,idx[1],idx[0]) = sol_in[new_base_addr + new_tIdx];  
    }

    if(ty == 0)
    {
      if(by == 0)
      {
        new_tIdx = tIdx;
        new_base_addr = base_addr;
      }
      else
      {
        new_tIdx = tIdx + (BLOCK_LENGTH-1)*BLOCK_LENGTH;
        new_base_addr = (block_idx - xgridlength)*blocksize;
      }

      SOL(idx[2],ty,idx[0]) = sol_in[new_base_addr + new_tIdx];
    }

    if(ty == BLOCK_LENGTH-1)
    {
      if(by == ygridlength-1) 
      {
        new_tIdx = tIdx;
        new_base_addr = base_addr;
      }
      else
      {
        new_tIdx = tIdx - (BLOCK_LENGTH-1)*BLOCK_LENGTH;
        new_base_addr = (block_idx + xgridlength)*blocksize;
      }

      SOL(idx[2],ty+2,idx[0]) = sol_in[new_base_addr + new_tIdx];
    }

    if(tz == 0)
    {
      if(bz == 0)
      {
        new_tIdx = tIdx;
        new_base_addr = base_addr;
      }
      else
      {
        new_tIdx = tIdx + (BLOCK_LENGTH-1)*BLOCK_LENGTH*BLOCK_LENGTH;
        new_base_addr = (block_idx - xgridlength*ygridlength)*blocksize;
      }

      SOL(idx[2],idx[1],tz) = sol_in[new_base_addr + new_tIdx];
    }

    if(tz == BLOCK_LENGTH-1)
    {
      if(bz == zgridlength-1) 
      {
        new_tIdx = tIdx;
        new_base_addr = base_addr;
      }
      else
      {
        new_tIdx = tIdx - (BLOCK_LENGTH-1)*BLOCK_LENGTH*BLOCK_LENGTH;
        new_base_addr = (block_idx + xgridlength*ygridlength)*blocksize;
      }

      SOL(idx[2],idx[1],tz+2) = sol_in[new_base_addr + new_tIdx];
    }

    __syncthreads();

    DOUBLE a,b,c,oldT,newT;

    for(int iter=0; iter<nIter; iter++)  
    {
      //
      // compute new value
      //
      oldT = newT = SOL(idx[2],idx[1],idx[0]);

      if(isValid)
      {
        a = sycl::min(SOL(tx,idx[1],idx[0]),SOL(tx+2,idx[1],idx[0]));
        b = sycl::min(SOL(idx[2],ty,idx[0]),SOL(idx[2],ty+2,idx[0]));
        c = sycl::min(SOL(idx[2],idx[1],tz),SOL(idx[2],idx[1],tz+2));

        DOUBLE tmp = (DOUBLE) get_time_eikonal(a, b, c, F);

        newT = sycl::min(tmp,oldT);
      }
      __syncthreads();  

      if(isValid) SOL(idx[2],idx[1],idx[0]) = newT;

      __syncthreads(); // this may not required    
    }

    DOUBLE residue = oldT - newT;

    // write back to global memory
    con[base_addr + tIdx] = (residue < EPS) ? true : false;
    sol_out[base_addr + tIdx] = newT;    
  }
}

SYCL_EXTERNAL
void run_reduction(
  sycl::nd_item<3> &item,
  bool *__restrict conv,
  const bool *__restrict con,
  bool *__restrict listVol,
  const uint *__restrict list,
  uint nActiveBlock)
{
  uint list_idx = item.get_group(1)*item.get_group_range(2) + item.get_group(2);

  if(list_idx < nActiveBlock)
  {
    uint block_idx = list[list_idx];

    uint blocksize = BLOCK_LENGTH*BLOCK_LENGTH*BLOCK_LENGTH/2;
    uint base_addr = block_idx*blocksize*2;
    uint tx = item.get_local_id(2);
    uint ty = item.get_local_id(1);
    uint tz = item.get_local_id(0);
    uint tIdx = tz*BLOCK_LENGTH*BLOCK_LENGTH + ty*BLOCK_LENGTH + tx;

    conv[tIdx] = con[base_addr + tIdx];
    conv[tIdx + blocksize] = con[base_addr + tIdx + blocksize];

    __syncthreads();

    for(uint i=blocksize; i>0; i/=2)
    {
      if(tIdx < i)
      {
        bool b1, b2;
        b1 = conv[tIdx];
        b2 = conv[tIdx+i];
        conv[tIdx] = (b1 && b2) ? true : false ;
      }
      __syncthreads();
    }

    if(tIdx == 0) 
    {    
      listVol[block_idx] = !conv[0]; // active list is negation of tile convergence (active = not converged)
    }
  }
}

SYCL_EXTERNAL
void run_check_neighbor(
  sycl::nd_item<3> &item,
  DOUBLE *__restrict sol_mem,
  const double*__restrict spd,
  const bool*__restrict mask,
  const DOUBLE *__restrict sol_in,
  DOUBLE *__restrict sol_out,
  bool *__restrict con,
  const uint*__restrict list,
  int xdim, int ydim, int zdim,
  uint nActiveBlock, uint nTotalBlock)
{
  uint list_idx = item.get_group(1)*item.get_group_range(2) + item.get_group(2);

  if(list_idx < nTotalBlock)
  {
    double F;
    bool isValid;
    DOUBLE (*_sol)[BLOCK_LENGTH+2][BLOCK_LENGTH+2] =
      reinterpret_cast<DOUBLE (*)[BLOCK_LENGTH+2][BLOCK_LENGTH+2]>(sol_mem);

    uint block_idx = list[list_idx];
    uint blocksize = BLOCK_LENGTH*BLOCK_LENGTH*BLOCK_LENGTH;
    uint base_addr = block_idx*blocksize;

    uint tx = item.get_local_id(2);
    uint ty = item.get_local_id(1);
    uint tz = item.get_local_id(0);
    uint tIdx = tz*BLOCK_LENGTH*BLOCK_LENGTH + ty*BLOCK_LENGTH + tx;

    if(list_idx < nActiveBlock) // copy value
    {
      sol_out[base_addr + tIdx] = sol_in[base_addr + tIdx];
    } 
    else
    {
      uint xgridlength = xdim/BLOCK_LENGTH;
      uint ygridlength = ydim/BLOCK_LENGTH;
      uint zgridlength = zdim/BLOCK_LENGTH;

      // compute block index
      uint bx = block_idx%xgridlength;
      uint tmpIdx = (block_idx - bx)/xgridlength;
      uint by = tmpIdx%ygridlength;
      uint bz = (tmpIdx-by)/ygridlength;

      // copy global to shared memory
      //dim3 idx(tx+1,ty+1,tz+1);
      sycl::range<3> idx (tz+1,ty+1,tx+1);

      _sol[idx[2]][idx[1]][idx[0]] = sol_in[base_addr + tIdx];
      F = spd[base_addr + tIdx];
      if(F > 0) F = 1.0/F;
      isValid = mask[base_addr + tIdx];

      uint new_base_addr, new_tIdx;

      // 1-neighborhood values
      if(tx == 0) 
      {
        if(bx == 0) // end of the grid
        {  
          new_tIdx = tIdx;
          new_base_addr = base_addr;
        }
        else
        {
          new_tIdx = tIdx + BLOCK_LENGTH-1;
          new_base_addr = (block_idx - 1)*blocksize;  
        }
        _sol[tx][idx[1]][idx[0]] = sol_in[new_base_addr + new_tIdx];  
      }

      if(tx == BLOCK_LENGTH-1)
      {
        if(bx == xgridlength-1) // end of the grid
        {
          new_tIdx = tIdx;
          new_base_addr = base_addr;
        }
        else
        {
          new_tIdx = tIdx - (BLOCK_LENGTH-1);
          new_base_addr = (block_idx + 1)*blocksize;  
        }
        _sol[tx+2][idx[1]][idx[0]] = sol_in[new_base_addr + new_tIdx];  
      }

      if(ty == 0)
      {
        if(by == 0)
        {
          new_tIdx = tIdx;
          new_base_addr = base_addr;
        }
        else
        {
          new_tIdx = tIdx + (BLOCK_LENGTH-1)*BLOCK_LENGTH;
          new_base_addr = (block_idx - xgridlength)*blocksize;
        }
        _sol[idx[2]][ty][idx[0]] = sol_in[new_base_addr + new_tIdx];
      }

      if(ty == BLOCK_LENGTH-1)
      {
        if(by == ygridlength-1) 
        {
          new_tIdx = tIdx;
          new_base_addr = base_addr;
        }
        else
        {
          new_tIdx = tIdx - (BLOCK_LENGTH-1)*BLOCK_LENGTH;
          new_base_addr = (block_idx + xgridlength)*blocksize;
        }
        _sol[idx[2]][ty+2][idx[0]] = sol_in[new_base_addr + new_tIdx];
      }

      if(tz == 0)
      {
        if(bz == 0)
        {
          new_tIdx = tIdx;
          new_base_addr = base_addr;
        }
        else
        {
          new_tIdx = tIdx + (BLOCK_LENGTH-1)*BLOCK_LENGTH*BLOCK_LENGTH;
          new_base_addr = (block_idx - xgridlength*ygridlength)*blocksize;
        }
        _sol[idx[2]][idx[1]][tz] = sol_in[new_base_addr + new_tIdx];
      }

      if(tz == BLOCK_LENGTH-1)
      {
        if(bz == zgridlength-1) 
        {
          new_tIdx = tIdx;
          new_base_addr = base_addr;
        }
        else
        {
          new_tIdx = tIdx - (BLOCK_LENGTH-1)*BLOCK_LENGTH*BLOCK_LENGTH;
          new_base_addr = (block_idx + xgridlength*ygridlength)*blocksize;
        }
        _sol[idx[2]][idx[1]][tz+2] = sol_in[new_base_addr + new_tIdx];
      }

      __syncthreads();


      DOUBLE a,b,c,oldT,newT;

      //
      // compute new value
      //
      oldT = newT = _sol[idx[2]][idx[1]][idx[0]];

      if(isValid)
      {
        a = sycl::min(_sol[tx][idx[1]][idx[0]],_sol[tx+2][idx[1]][idx[0]]);
        b = sycl::min(_sol[idx[2]][ty][idx[0]],_sol[idx[2]][ty+2][idx[0]]);
        c = sycl::min(_sol[idx[2]][idx[1]][tz],_sol[idx[2]][idx[1]][tz+2]);

        DOUBLE tmp = (DOUBLE) get_time_eikonal(a, b, c, F);
        newT = sycl::min(tmp,oldT);

        sol_out[base_addr + tIdx] = newT;
      }
      // write back to global memory
      DOUBLE residue = oldT - newT;
      con[base_addr + tIdx] = (residue < EPS) ? true : false;  
    }
  }
}

