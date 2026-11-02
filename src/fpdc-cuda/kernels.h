#define ull unsigned long long
#define MAX (64*1024*1024)
#define WARPSIZE 32


/************************************************************************************/

/*
This is the GPU compression kernel, which should be launched using the block count
and warps/block:
  CompressionKernel<<<blocks, WARPSIZE*warpsperblock>>>();

Inputs
------
dimensionalityd: dimensionality of trace (from cmd line)
cbufd: ptr to the uncompressed data
cutd: ptr to array of chunk boundaries

Output
------
The compressed data, in dbufd 
Compressed chunk offsets for offset table, in offd
*/

__global__ void CompressionKernel(
  const int dimensionalityd,
  const ull *__restrict__ cbufd,
  char *__restrict__ dbufd,
  const int *__restrict__ cutd,
  int *__restrict__ offd)
{
  int offset, code, bcount, tmp, off, beg, end, lane, warp, iindex, lastidx, start, term;
  ull diff, prev;
  code = 0;
  __shared__ int ibufs[32 * (3 * WARPSIZE / 2)]; // shared space for prefix sum

  // index within this warp
  lane = threadIdx.x & 31;
  // index within shared prefix sum array
  iindex = threadIdx.x / WARPSIZE * (3 * WARPSIZE / 2) + lane;
  ibufs[iindex] = 0;
  iindex += WARPSIZE / 2;
  lastidx = (threadIdx.x / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;
  // warp id
  warp = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
  // prediction index within previous subchunk
  offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

  // determine start and end of chunk to compress
  start = 0;
  if (warp > 0) start = cutd[warp-1];
  term = cutd[warp];
  off = ((start+1)/2*17);

  prev = 0;
  for (int i = start + lane; i < term; i += WARPSIZE) {
    // calculate delta between value to compress and prediction
    // and negate if negative
    if (cbufd[i] >= prev)
      diff = cbufd[i] - prev;
    else
      diff = prev - cbufd[i];

    // count leading zeros in positive delta
    bcount = 8 - (__clzll(diff) >> 3);
    if (bcount == 2) bcount = 3; // encode 6 lead-zero bytes as 5

    // prefix sum to determine start positions of non-zero delta bytes
    ibufs[iindex] = bcount;
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-1];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-2];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-4];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-8];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-16];
    __threadfence_block();

    // write out non-zero bytes of delta to compressed buffer
    beg = off + (WARPSIZE/2) + ibufs[iindex-1];
    end = beg + bcount;
    for (; beg < end; beg++) {
      dbufd[beg] = diff;
      diff >>= 8;
    }

    if (bcount >= 3) bcount--; // adjust byte count for the dropped encoding
    tmp = ibufs[lastidx];
    code |= bcount;
    ibufs[iindex] = code;
    __threadfence_block();

    // write out half-bytes of sign and leading-zero-byte count (every other thread
    // writes its half-byte and neighbor's half-byte)
    if ((lane & 1) != 0) {
      dbufd[off + (lane >> 1)] = ibufs[iindex-1] | (code << 4);
    }
    off += tmp + (WARPSIZE/2);

    // save prediction value from this subchunk (based on provided dimensionality)
    // for use in next subchunk
    prev = cbufd[i + offset];
  }

  // save final value of off, which is total bytes of compressed output for this chunk
  if (lane == 31) offd[warp] = off;
}
