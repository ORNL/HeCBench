//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  trueke // A multi-GPU implementation of the exchange Monte Carlo method. //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang. //
//                                                                              //
//  This file is part of trueke. // trueke is free software: you can
//  redistribute it and/or modify              // it under the terms of the GNU
//  General Public License as published by        // the Free Software
//  Foundation, either version 3 of the License, or           // (at your
//  option) any later version.                                         //
//                                                                              //
//  trueke is distributed in the hope that it will be useful, // but WITHOUT ANY
//  WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the // GNU General
//  Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License // along
//  with trueke.  If not, see <http://www.gnu.org/licenses/>.             //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////

#ifndef _KERNEL_MONTECARLO_CUH_
#define _KERNEL_MONTECARLO_CUH_

// GPU-related definitions

#define sLx (BX+2)
#define sLy (BY+2)
#define sLz (BZ+2)

#define SVOLUME ((sLx)*(sLy)*(sLz))
#define BVOLUME ((BX)*((BY)/2)*(BZ))

#define BLOCKSIZE 8
#define BLOCK_STEPS 1
#define EPSILON 0.00000000001f

#define C(x,y,z,L)     ((z)*(L)*(L)+(y)*(L)+(x))
#define sC(x,y,z,Lx,Ly)  ((z+1)*(Ly)*(Lx)+(y+1)*(Lx)+(x+1))

typedef int site_t;

void
kernel_metropolis(const int N, const int L, site_t *s, const int *H, 
                  const float h, const float B, uint64_t *state, uint64_t *inc, int alt,
                  sycl::nd_item<3> &item, site_t *ss)
{
  // offsets
  int offx = item.get_group(2) * BX;
  int offy = (2 * item.get_group(1) +
              ((item.get_group(2) + item.get_group(0) + alt) & 1)) *
             BY;
  int offz = item.get_group(0) * BZ;

  // halo shared memory coords
  int sx = item.get_local_id(2);
  int sy1 = 2 * item.get_local_id(1);
  int sy2 = 2 * item.get_local_id(1) + 1;
  int sz = item.get_local_id(0);

  // global coords
  int x = offx + sx;
  int y1 = offy + sy1;
  int y2 = offy + sy2;
  int z = offz + sz;

  //if(x >= N || y1 >= N || y2 >= N || z >= N)
  //return;

  // global and local and block id in soc
  int tid = z * L * L / 4 +
            (item.get_group(1) * BY / 2 + item.get_local_id(1)) * L + x;

  // load the spins into shared memory
  ss[sC(sx, sy1, sz, sLx, sLy)] = s[C(x, y1, z, L)];
  ss[sC(sx, sy2, sz, sLx, sLy)] = s[C(x, y2, z, L)];
  // get the h1,h2 values for y1 y2.
  int h1 = H[C(x, y1, z, L)];
  int h2 = H[C(x, y2, z, L)];
  //printf("thread %i   h1=%i   h2=%i\n", tid, h1, h2);

  // ------------------------------------------------
  // halo
  // ------------------------------------------------
  // Y boundary
  if (item.get_local_id(1) == 0) {
    // we also check if we are on the limit of the lattice
    ss[sC(sx, -1, sz, sLx, sLy)] = (offy == 0) ? s[C(x, L-1, z, L)] : s[C(x, offy-1, z, L)];
  }
  if (item.get_local_id(1) == BY / 2 - 1) {
    ss[sC(sx, BY, sz, sLx, sLy)] = (offy == L-BY) ? s[C(x, 0, z, L)] : s[C(x, offy+BY, z, L)];
  }

  // X boundary
  if (item.get_local_id(2) == 0) {
    if (item.get_group(2) == 0) {
      ss[sC(-1, sy1, sz, sLx, sLy)] = s[C(L-1, y1, z, L)];
      ss[sC(-1, sy2, sz, sLx, sLy)] = s[C(L-1, y2, z, L)];
    }
    else{
      ss[sC(-1, sy1, sz, sLx, sLy)] = s[C(offx-1, y1, z, L)];
      ss[sC(-1, sy2, sz, sLx, sLy)] = s[C(offx-1, y2, z, L)];
    }
  }
  if (item.get_local_id(2) == BX - 1) {
    if (item.get_group(2) == item.get_group_range(2) - 1) {
      ss[sC(BX, sy1, sz, sLx, sLy)] = s[C(0, y1, z, L)];
      ss[sC(BX, sy2, sz, sLx, sLy)] = s[C(0, y2, z, L)];
    }
    else{
      ss[sC(BX, sy1, sz, sLx, sLy)] = s[C(offx+BX, y1, z, L)];
      ss[sC(BX, sy2, sz, sLx, sLy)] = s[C(offx+BX, y2, z, L)];
    }
  }

  // Z boundary
  if (item.get_local_id(0) == 0) {
    if (item.get_group(0) == 0) {
      ss[sC(sx, sy1, -1, sLx, sLy)] = s[C(x, y1, L-1, L)];
      ss[sC(sx, sy2, -1, sLx, sLy)] = s[C(x, y2, L-1, L)];
    }
    else{
      ss[sC(sx, sy1, -1, sLx, sLy)] = s[C(x, y1, offz-1, L)];
      ss[sC(sx, sy2, -1, sLx, sLy)] = s[C(x, y2, offz-1, L)];
    }
  }
  if (item.get_local_id(0) == BZ - 1) {
    if (item.get_group(0) == item.get_group_range(0) - 1) {
      ss[sC(sx, sy1, BZ, sLx, sLy)] = s[C(x, y1, 0, L)];
      ss[sC(sx, sy2, BZ, sLx, sLy)] = s[C(x, y2, 0, L)];
    }
    else{
      ss[sC(sx, sy1, BZ, sLx, sLy)] = s[C(x, y1, offz+BZ, L)];
      ss[sC(sx, sy2, BZ, sLx, sLy)] = s[C(x, y2, offz+BZ, L)];
    }
  }

  // get random number state
  uint64_t lstate = state[tid];
  uint64_t linc = inc[tid];
  // the white and black y
  int wy = ((sx + sz) & 1) + 2 * item.get_local_id(1);
  int by = ((sx + sz + 1) & 1) + 2 * item.get_local_id(1);
  float dh;
  int c;

  item.barrier(sycl::access::fence_space::local_space);

  for(int i = 0; i < BLOCK_STEPS; ++i){

    /* -------- white update -------- */
    dh = (float)(ss[sC(sx, wy, sz, sLx, sLy)] * (
         (float)(ss[sC(sx-1,wy,sz, sLx, sLy)] + ss[sC(sx+1, wy, sz, sLx, sLy)] + 
                 ss[sC(sx,wy-1,sz, sLx, sLy)] + ss[sC(sx, wy+1, sz, sLx, sLy)] +
                 ss[sC(sx,wy,sz-1, sLx, sLy)] + ss[sC(sx, wy, sz+1, sLx, sLy)]) + h*h1));
    c = sycl::signbit(dh - EPSILON) |
        sycl::signbit(gpu_rand01(&lstate, &linc) - sycl::exp(dh * B));
    ss[sC(sx, wy, sz, sLx, sLy)] *= (1 - 2*c);

    item.barrier(sycl::access::fence_space::local_space);

    /* -------- black update -------- */
    dh = (float)(ss[sC(sx, by, sz, sLx, sLy)] * (
         (float)(ss[sC(sx-1,by,sz, sLx, sLy)] + ss[sC(sx+1, by, sz, sLx, sLy)] + 
                 ss[sC(sx,by-1,sz, sLx, sLy)] + ss[sC(sx, by+1, sz, sLx, sLy)] +
                 ss[sC(sx,by,sz-1, sLx, sLy)] + ss[sC(sx, by, sz+1, sLx, sLy)]) + h*h2));

    c = sycl::signbit(dh - EPSILON) |
        sycl::signbit(gpu_rand01(&lstate, &linc) - sycl::exp(dh * B));
    ss[sC(sx, by, sz, sLx, sLy)] *= (1 - 2*c);

    item.barrier(sycl::access::fence_space::local_space);
  }

  /* copy data back to gmem */
  s[C(x, y1, z, L)] = ss[sC(sx, sy1, sz, sLx, sLy)];
  s[C(x, y2, z, L)] = ss[sC(sx, sy2, sz, sLx, sLy)]; 
  /* update random number state */
  state[tid] = lstate;
  inc[tid] = linc;
}


// NOTE: the space of computation is 1/4 of N, so that is why each thread does quadruple work.
void 
kernel_reset_random_gpupcg(int *s, int N, uint64_t *state, uint64_t *inc,
                           sycl::nd_item<1> &item)
{
  int x = item.get_global_id(0);
  float v;
  /* Each thread gets same seed, a different sequence number, no offset */
  if( x >= N/4 ) return;

  /* get the prng state in register memory */
  uint64_t lstate = state[x];
  uint64_t linc = inc[x];

  // first random
  v = (int) (gpu_rand01(&lstate, &linc) + 0.5f);
  s[x] = 1-2*v;
  // second random
  v = (int) (gpu_rand01(&lstate, &linc) + 0.5f);
  s[x + N/4] = 1-2*v;
  // third random
  v = (int) (gpu_rand01(&lstate, &linc) + 0.5f);
  s[x + N/2 ]  = 1-2*v;
  // fourth random
  v = (int) (gpu_rand01(&lstate, &linc) + 0.5f);
  s[x + 3*N/4] = 1-2*v;

  /* save the state back to global memory */
  state[x] = lstate;
  inc[x] = linc;
}

template<typename T>
void kernel_reset(T *a, int N, T val, sycl::nd_item<1> &item){
  int idx = item.get_global_id(0);
  if(idx < N) a[idx] = val;
}
#endif
