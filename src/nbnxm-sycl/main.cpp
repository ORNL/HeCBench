#include <iostream>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "vectypes.h"

typedef sycl::float2 Float2;
typedef gmx::BasicVector<float> Float3;
typedef sycl::float4 Float4;

#include "constants.h"
#include "reference.h"

template<typename T>
static inline void atomicAdd(T& val, const T delta)
{
  sycl::atomic_ref<T, sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ref(val);
  ref.fetch_add(delta);
}

static __attribute__((always_inline)) float pmeCorrF(const float z2)
{
  constexpr float FN6 = -1.7357322914161492954e-8F;
  constexpr float FN5 = 1.4703624142580877519e-6F;
  constexpr float FN4 = -0.000053401640219807709149F;
  constexpr float FN3 = 0.0010054721316683106153F;
  constexpr float FN2 = -0.019278317264888380590F;
  constexpr float FN1 = 0.069670166153766424023F;
  constexpr float FN0 = -0.75225204789749321333F;

  constexpr float FD4 = 0.0011193462567257629232F;
  constexpr float FD3 = 0.014866955030185295499F;
  constexpr float FD2 = 0.11583842382862377919F;
  constexpr float FD1 = 0.50736591960530292870F;
  constexpr float FD0 = 1.0F;

  const float z4 = z2 * z2;
        float polyFD0 = FD4 * z4 + FD2;
  const float polyFD1 = FD3 * z4 + FD1;
  polyFD0 = polyFD0 * z4 + FD0;
  polyFD0 = polyFD1 * z2 + polyFD0;
  polyFD0 = 1.0F / polyFD0;

  float polyFN0 = FN6 * z4 + FN4;
  float polyFN1 = FN5 * z4 + FN3;
  polyFN0       = polyFN0 * z4 + FN2;
  polyFN1       = polyFN1 * z4 + FN1;
  polyFN0       = polyFN0 * z4 + FN0;
  polyFN0       = polyFN1 * z2 + polyFN0;

  return polyFN0 * polyFD0;
}

static __attribute__((always_inline))
  void reduceForceIAndFShiftXYZ(
      float* sm_buf,
      const float* fCiBufX,
      const float* fCiBufY,
      const float* fCiBufZ,
      const bool calcFShift,
      const sycl::nd_item<3> itemIdx,
      const int tidxi,
      const int tidxj,
      const int sci,
      const int shift,
      sycl::global_ptr<Float3> a_f,
      sycl::global_ptr<Float3> a_fShift) {

    static constexpr int bufStride  = c_clSize * c_clSize;
    static constexpr int clSizeLog2 = StaticLog2<c_clSize>::value;
    const int            tidx       = tidxi + tidxj * c_clSize;
    float                fShiftBuf  = 0.0F;
#pragma unroll(8)
    for (int ciOffset = 0; ciOffset < c_nbnxnGpuNumClusterPerSupercluster; ciOffset++)
    {
      const int aidx = (sci * c_nbnxnGpuNumClusterPerSupercluster + ciOffset) * c_clSize + tidxi;
      /* store i forces in shmem */
      sm_buf[tidx]                 = fCiBufX[ciOffset];
      sm_buf[bufStride + tidx]     = fCiBufY[ciOffset];
      sm_buf[2 * bufStride + tidx] = fCiBufZ[ciOffset];
      itemIdx.barrier(sycl::access::fence_space::local_space);

      /* Reduce the initial c_clSize values for each i atom to half
       * every step by using c_clSize * i threads. */
      int i = c_clSize / 2;
      for (int j = clSizeLog2 - 1; j > 0; j--)
      {
        if (tidxj < i)
        {
          sm_buf[tidx] += sm_buf[tidx + i * c_clSize];
          sm_buf[bufStride + tidx] += sm_buf[bufStride + tidx + i * c_clSize];
          sm_buf[2 * bufStride + tidx] += sm_buf[2 * bufStride + tidx + i * c_clSize];
        }
        i >>= 1;
        itemIdx.barrier(sycl::access::fence_space::local_space);
      }

      /* i == 1, last reduction step, writing to global mem */
      /* Split the reduction between the first 3 line threads
         Threads with line id 0 will do the reduction for (float3).x components
         Threads with line id 1 will do the reduction for (float3).y components
         Threads with line id 2 will do the reduction for (float3).z components. */
      if (tidxj < 3)
      {
        const float f =
          sm_buf[tidxj * bufStride + tidxi] + sm_buf[tidxj * bufStride + c_clSize + tidxi];
        atomicAdd(a_f[aidx][tidxj], f);
        if (calcFShift)
        {
          fShiftBuf += f;
        }
      }
      itemIdx.barrier(sycl::access::fence_space::local_space);
    }
    /* add up local shift forces into global mem */
    if (calcFShift)
    {
      /* Only threads with tidxj < 3 will update fshift.
         The threads performing the update must be the same as the threads
         storing the reduction result above. */
      if (tidxj < 3)
      {
        if constexpr (c_clSize == 4)
        {
          /* Intel Xe (Gen12LP) and earlier GPUs implement floating-point atomics via
           * a compare-and-swap (CAS) loop. It has particularly poor performance when
           * updating the same memory location from the same work-group.
           * Such optimization might be slightly beneficial for NVIDIA and AMD as well,
           * but it is unlikely to make a big difference and thus was not evaluated.
           */
          auto sg = itemIdx.get_sub_group();
          fShiftBuf += sycl::shift_group_left(sg,fShiftBuf, 1);
          fShiftBuf += sycl::shift_group_left(sg,fShiftBuf, 2);
          if (tidxi == 0)
          {
            atomicAdd(a_fShift[shift][tidxj], fShiftBuf);
          }
        }
        else
        {
          atomicAdd(a_fShift[shift][tidxj], fShiftBuf);
        }
      }
    }
  }

static __attribute__((always_inline)) void reduceForceJShuffle(Float3                                   f,
    const sycl::nd_item<3> itemIdx,
    const int tidxi,
    const int aidx,
    sycl::global_ptr<Float3> a_f)
{
  static_assert(c_clSize == 8 || c_clSize == 4);
  auto sg = itemIdx.get_sub_group();

  f[0] += sycl::shift_group_left(sg,f[0], 1);
  f[1] += sycl::shift_group_right(sg,f[1], 1);
  f[2] += sycl::shift_group_left(sg,f[2], 1);
  if (tidxi & 1)
  {
    f[0] = f[1];
  }

  f[0] += sycl::shift_group_left(sg,f[0], 2);
  f[2] += sycl::shift_group_right(sg,f[2], 2);
  if (tidxi & 2)
  {
    f[0] = f[2];
  }

  if constexpr (c_clSize == 8)
  {
    f[0] += sycl::shift_group_left(sg,f[0], 4);
  }

  if (tidxi < 3)
  {
    atomicAdd(a_f[aidx][tidxi], f[0]);
  }
}

static __attribute__((always_inline)) void reduceForceJ(
    float* sm_buf,
    Float3 f,
    const sycl::nd_item<3> itemIdx,
    const int tidxi,
    const int tidxj,
    const int aidx,
    const sycl::global_ptr<Float3> a_f)
{
  reduceForceJShuffle(f, itemIdx, tidxi, aidx, a_f);
}

auto nbnxmKernelTest(
    sycl::handler& cgh,
    const Float4 *a_xq,
    Float3 *a_f,
    Float3 *a_shiftVec,
    Float3 *a_fShift,
    nbnxn_cj4_t *a_plistCJ4,
    const nbnxn_sci_t *a_plistSci,
    const nbnxn_excl_t *a_plistExcl,
    const int *a_atomTypes,
    const Float2 *a_nbfp,
    const int numTypes,
    const float rCoulombSq,
    const float ewaldBeta,
    const float epsFac,
    const bool calcShift) {
  return [=](sycl::nd_item<3> itemIdx) {

    constexpr int prunedClusterPairSize = c_clSize * c_splitClSize;

    /* thread/block/warp id-s */
    const unsigned tidxi = itemIdx.get_local_id(2);
    const unsigned tidxj = itemIdx.get_local_id(1);
    const unsigned tidx  = tidxj * c_clSize + tidxi;
    const unsigned bidx = itemIdx.get_group(0);

    const unsigned imeiIdx = tidx / prunedClusterPairSize;

    constexpr size_t local_mem_size =
      c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(Float4) + // sm_xq
      c_clSize * c_clSize * DIM * sizeof(float) +                       // sm_reductionBuffer
      c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(int);     // sm_atomTypeI

    auto group = itemIdx.get_group();
    uint8_t *ptr =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t[local_mem_size]>(group);

    Float4* sm_xq = reinterpret_cast<Float4*>(ptr);
    ptr += c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(Float4);
    float* sm_reductionBuffer = reinterpret_cast<float*>(ptr);
    ptr += c_clSize * c_clSize * DIM * sizeof(float);
    int* sm_atomTypeI = reinterpret_cast<int*>(ptr);

    float fCiBufX[c_nbnxnGpuNumClusterPerSupercluster] = {0};
    float fCiBufY[c_nbnxnGpuNumClusterPerSupercluster] = {0};
    float fCiBufZ[c_nbnxnGpuNumClusterPerSupercluster] = {0};

    const nbnxn_sci_t nbSci     = a_plistSci[bidx];
    const int         sci       = nbSci.sci;
    const int         cij4Start = nbSci.cj4_ind_start;
    const int         cij4End   = nbSci.cj4_ind_end;
    const int         nbScishift = nbSci.shift;

    // Only needed if props.elecEwaldAna
    const float beta2 = ewaldBeta * ewaldBeta;
    const float beta3 = ewaldBeta * ewaldBeta * ewaldBeta;

    for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i += c_clSize)
    {
      /* Pre-load i-atom x and q into shared memory */
      const int ci = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxj + i;
      const int ai = ci * c_clSize + tidxi;

      //const Float3 shift = a_shiftVec[nbSci.shift];
      const Float3 shift = a_shiftVec[nbScishift];
      Float4       xqi   = a_xq[ai];
      xqi += Float4(shift[0], shift[1], shift[2], 0.0F);
      xqi[3] *= epsFac;
      sm_xq[(tidxj + i) * c_clSize + tidxi] = xqi;

      sm_atomTypeI[(tidxj + i) * c_clSize + tidxi] = a_atomTypes[ai];
    }
    itemIdx.barrier(sycl::access::fence_space::local_space);

    // Only needed if (doExclusionForces)
    const bool nonSelfInteraction = !(nbScishift == c_centralShiftIndex & tidxj <= tidxi);

    // loop over the j clusters = seen by any of the atoms in the current super-cluster
    for (int j4 = cij4Start; j4 < cij4End; j4 += 1)
    {
      unsigned imask = a_plistCJ4[j4].imei[imeiIdx].imask;
      if (!imask)
      {
        continue;
      }
      const int wexclIdx = a_plistCJ4[j4].imei[imeiIdx].excl_ind;
      const unsigned wexcl = a_plistExcl[wexclIdx].pair[tidx & (prunedClusterPairSize - 1)];

      for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
      {
        const bool maskSet =
          imask & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster));
        if (!maskSet)
        {
          continue;
        }
        unsigned  maskJI = (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster));
        const int cj     = a_plistCJ4[j4].cj[jm];
        const int aj     = cj * c_clSize + tidxj;

        // load j atom data
        const Float4 xqj = a_xq[aj];

        const Float3 xj(xqj[0], xqj[1], xqj[2]);
        const float  qj = xqj[3];
        int          atomTypeJ; // Only needed if (!props.vdwComb)
        atomTypeJ = a_atomTypes[aj];

        Float3 fCjBuf(0.0F, 0.0F, 0.0F);

#pragma unroll(8)
        for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
        {
          if (imask & maskJI)
          {
            // i cluster index
            const int ci = sci * c_nbnxnGpuNumClusterPerSupercluster + i;
            // all threads load an atom from i cluster ci into shmem!
            const Float4 xqi = sm_xq[i * c_clSize + tidxi];
            const Float3 xi(xqi[0], xqi[1], xqi[2]);

            // distance between i and j atoms
            const Float3 rv = xi - xj;
            float        r2 = norm2(rv);

            const float pairExclMask = (wexcl & maskJI) ? 1.0F : 0.0F;

            // cutoff & exclusion check

            const bool notExcluded = (nonSelfInteraction | (ci != cj));

            // Check optimal way of branching here.
            if ((r2 < rCoulombSq) && notExcluded)
            {
              const float qi = xqi[3];
              int         atomTypeI; // Only needed if (!props.vdwComb)
              Float2      c6c12;

              /* LJ 6*C6 and 12*C12 */
              atomTypeI = sm_atomTypeI[i * c_clSize + tidxi];
              c6c12     = a_nbfp[numTypes * atomTypeI + atomTypeJ];

              // c6 and c12 are unused and garbage iff props.vdwCombLB && !doCalcEnergies
              const float c6  = c6c12[0];
              const float c12 = c6c12[1];

              // Ensure distance do not become so small that r^-12 overflows
              r2 = std::max(r2, c_nbnxnMinDistanceSquared);
              const float rInv = sycl::native::rsqrt(r2);
              const float r2Inv = rInv * rInv;
              float       r6Inv, fInvR;
              r6Inv = r2Inv * r2Inv * r2Inv;
              r6Inv *= pairExclMask;
              fInvR = r6Inv * (c12 * r6Inv - c6) * r2Inv;

              fInvR += qi * qj
                * (pairExclMask * r2Inv * rInv + pmeCorrF(beta2 * r2) * beta3);

              const Float3 forceIJ = rv * fInvR;

              /* accumulate j forces in registers */
              fCjBuf -= forceIJ;
              /* accumulate i forces in registers */
              fCiBufX[i] += forceIJ[0];
              fCiBufY[i] += forceIJ[1];
              fCiBufZ[i] += forceIJ[2];
            } // (r2 < rCoulombSq) && notExcluded
          }     // (imask & maskJI)
          /* shift the mask bit by 1 */
          maskJI += maskJI;
        } // for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
        /* reduce j forces */
        reduceForceJ(sm_reductionBuffer, fCjBuf, itemIdx, tidxi, tidxj, aj, a_f);
      } // for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
    } // for (int j4 = cij4Start; j4 < cij4End; j4 += 1)

    {
      const nbnxn_sci_t nbSci     = a_plistSci[itemIdx.get_group(0)];
      const int         sci       = nbSci.sci;

      // skip central shifts when summing shift forces
      const bool doCalcShift = (calcShift && nbSci.shift != c_centralShiftIndex);

      reduceForceIAndFShiftXYZ(
          sm_reductionBuffer, fCiBufX, fCiBufY, fCiBufZ, doCalcShift, itemIdx, tidxi, tidxj, sci, nbSci.shift, a_f, a_fShift);
    }
  };
}

nbnxn_cj4_t get_cj4(int id) {
  nbnxn_cj4_t value;
  for (int i = 0; i < c_nbnxnGpuJgroupSize; ++i) {
    value.cj[i] = i + id;
  }
  for (int i = 0; i < c_nbnxnGpuClusterpairSplit; ++i) {
    value.imei[i].imask = 0xFFFFFFFFu;
    value.imei[i].excl_ind = id % 19205;
  }
  return value;
}

nbnxn_excl_t get_excl(int id) {
  nbnxn_excl_t value;
  std::mt19937 rng(id);
  for (int i = 0; i < c_nbnxnGpuExclSize; ++i) value.pair[i] = rng();
  return value;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());

  const sycl::range<3>    blockSize{ 1, block_y, block_x };
  const sycl::range<3>    globalSize{ grid_z, block_y, block_x };
  const sycl::nd_range<3> range{ globalSize, blockSize };

  Float4*        h_xq        = new Float4[NUM_ATOMS];
  Float3*        h_f         = new Float3[NUM_ATOMS];
  Float3*        h_shiftVec  = new Float3[45];
  Float3*        h_fShift    = new Float3[45];
  nbnxn_cj4_t*   h_cj4       = new nbnxn_cj4_t[56881];
  nbnxn_sci_t*   h_sci       = new nbnxn_sci_t[4806];
  nbnxn_excl_t*  h_excl      = new nbnxn_excl_t[19205];
  int*           h_atomTypes = new int[NUM_ATOMS];
  Float2*        h_nbfp      = new Float2[1024];

  Float4*        d_xq;
  Float3*        d_f;
  Float3*        d_shiftVec;
  Float3*        d_fShift;
  nbnxn_cj4_t*   d_cj4;
  nbnxn_sci_t*   d_sci;
  nbnxn_excl_t*  d_excl;
  int*           d_atomTypes;
  Float2*        d_nbfp;

  d_xq        = sycl::malloc_device<Float4>(NUM_ATOMS, q);
  d_f         = sycl::malloc_device<Float3>(NUM_ATOMS, q);
  d_shiftVec  = sycl::malloc_device<Float3>(45, q);
  d_fShift    = sycl::malloc_device<Float3>(45, q);
  d_cj4       = sycl::malloc_device<nbnxn_cj4_t>(56881, q);
  d_sci       = sycl::malloc_device<nbnxn_sci_t>(4806, q);
  d_excl      = sycl::malloc_device<nbnxn_excl_t>(19205, q);
  d_atomTypes = sycl::malloc_device<int>(NUM_ATOMS, q);
  d_nbfp      = sycl::malloc_device<Float2>(1024, q);

  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> posDist(-20.f, 20.f);
  std::uniform_real_distribution<float> qDist(-1.f, 1.f);

  for (int i = 0; i < NUM_ATOMS; ++i) {
    h_xq[i] = Float4(posDist(rng), posDist(rng), posDist(rng), qDist(rng));
  }
  for (int i = 0; i < NUM_ATOMS; ++i) {
    h_f[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 45; ++i) {
    h_shiftVec[i] = Float3(posDist(rng)*0.1f, posDist(rng)*0.1f, posDist(rng)*0.1f);
  }
  for (int i = 0; i < 45; ++i) {
    h_fShift[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 56881; ++i) {
    h_cj4[i] = get_cj4(i % 200);
  }
  for (int i = 0; i < 4806; ++i) {
    h_sci[i] = {i % 400, i % c_numIvecs, (2*i) % 200, (2*i) % 200 + 1};
  }
  for (int i = 0; i < 19205; ++i) {
    h_excl[i] = get_excl(i);
  }
  for (int i = 0; i < NUM_ATOMS; ++i) {
    h_atomTypes[i] = (i % 32);
  }
  for (int i = 0; i < 1024; ++i) {
    h_nbfp[i] = Float2(0.5f, 0.25f);
  }

  q.memcpy(d_xq,        h_xq,        sizeof(Float4)       * NUM_ATOMS);
  q.memcpy(d_f,         h_f,         sizeof(Float3)       * NUM_ATOMS);
  q.memcpy(d_shiftVec,  h_shiftVec,  sizeof(Float3)       * 45);
  q.memcpy(d_fShift,    h_fShift,    sizeof(Float3)       * 45);
  q.memcpy(d_cj4,       h_cj4,       sizeof(nbnxn_cj4_t)  * 56881);
  q.memcpy(d_sci,       h_sci,       sizeof(nbnxn_sci_t)  * 4806);
  q.memcpy(d_excl,      h_excl,      sizeof(nbnxn_excl_t) * 19205);
  q.memcpy(d_atomTypes, h_atomTypes, sizeof(int)          * NUM_ATOMS);
  q.memcpy(d_nbfp,      h_nbfp,      sizeof(Float2)       * 1024);

  // NbnxmReference reads its inputs directly from host memory -- pass the
  // h_* buffers, not the d_* ones (it cannot dereference device pointers).
  NbnxmReference ref(h_xq, h_shiftVec, h_cj4, h_sci, h_excl, h_atomTypes, h_nbfp,
                     32, 1, 3.12341f, 138.935f);

  // Warming-up
  q.submit([&](sycl::handler& cgh) {
      auto kernel = nbnxmKernelTest(
          cgh,
          d_xq,
          d_f,
          d_shiftVec,
          d_fShift,
          d_cj4,
          d_sci,
          d_excl,
          d_atomTypes,
          d_nbfp,
          32,
          1,
          3.12341,
          138.935,
          0);
      cgh.parallel_for(range, kernel);
  });
  q.wait();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    q.submit([&](sycl::handler& cgh) {
        auto kernel = nbnxmKernelTest(
            cgh,
            d_xq,
            d_f,
            d_shiftVec,
            d_fShift,
            d_cj4,
            d_sci,
            d_excl,
            d_atomTypes,
            d_nbfp,
            32,
            1,
            3.12341,
            138.935,
            0);
        cgh.parallel_for(range, kernel);
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (w/o shift): %f (us)\n", (time * 1e-3f) / repeat);

  // Pull the accumulated forces back to host for validation against the
  // CPU reference (d_f / d_fShift are device-only USM, not host-readable).
  q.memcpy(h_f,      d_f,      sizeof(Float3) * NUM_ATOMS);
  q.memcpy(h_fShift, d_fShift, sizeof(Float3) * 45);
  q.wait();

  ref.computeDelta(/*calcShift=*/false);
  ref.validate(h_f, h_fShift, /*launchCount=*/repeat + 1,
               1.0f, 0.5f, 0.25f, 1.0f, 0.5f, 0.25f,
               /*absTol=*/1e-3f, "w/o shift");

  // Reset the force accumulators (host copies), then push the reset back
  // to device before the second (w/ shift) timed block, exactly mirroring
  // what the original sycl::malloc_shared version did in-place.
  for (int i = 0; i < NUM_ATOMS; ++i) {
    h_f[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 45; ++i) {
    h_fShift[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  q.memcpy(d_f,      h_f,      sizeof(Float3) * NUM_ATOMS);
  q.memcpy(d_fShift, h_fShift, sizeof(Float3) * 45);
  q.wait();

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; ++i) {
    q.submit([&](sycl::handler& cgh) {
        auto kernel = nbnxmKernelTest(
            cgh,
            d_xq,
            d_f,
            d_shiftVec,
            d_fShift,
            d_cj4,
            d_sci,
            d_excl,
            d_atomTypes,
            d_nbfp,
            32,
            1,
            3.12341,
            138.935,
            1);
        cgh.parallel_for(range, kernel);
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (w/ shift): %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(h_f,      d_f,      sizeof(Float3) * NUM_ATOMS);
  q.memcpy(h_fShift, d_fShift, sizeof(Float3) * 45);
  q.wait();

  ref.computeDelta(/*calcShift=*/true);
  // Second block's `repeat` launches all pass calcShift=1; there is
  // no extra warm-up launch before this block.
  ref.validate(h_f, h_fShift, /*launchCount=*/repeat,
               1.0f, 0.5f, 0.25f, 1.0f, 0.5f, 0.25f,
               /*absTol=*/1e-3f, "w/ shift");

  sycl::free(d_nbfp, q);
  sycl::free(d_atomTypes, q);
  sycl::free(d_excl, q);
  sycl::free(d_sci, q);
  sycl::free(d_cj4, q);
  sycl::free(d_fShift, q);
  sycl::free(d_shiftVec, q);
  sycl::free(d_f, q);
  sycl::free(d_xq, q);

  delete[] h_nbfp;
  delete[] h_atomTypes;
  delete[] h_excl;
  delete[] h_sci;
  delete[] h_cj4;
  delete[] h_fShift;
  delete[] h_shiftVec;
  delete[] h_f;
  delete[] h_xq;

  return 0;
}

