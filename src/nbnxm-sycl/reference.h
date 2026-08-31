#pragma once

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

static inline float pmeCorrF_host(const float z2)
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

struct HostVec3 { float x = 0.f, y = 0.f, z = 0.f; };

class NbnxmReference
{
public:
  NbnxmReference(const Float4*        xq,
                 const Float3*        shiftVec,
                 const nbnxn_cj4_t*   cj4,
                 const nbnxn_sci_t*   sci,
                 const nbnxn_excl_t*  excl,
                 const int*           atomTypes,
                 const Float2*        nbfp,
                 int                  numTypes,
                 float                rCoulombSq,
                 float                ewaldBeta,
                 float                epsFac)
    : xq_(xq), shiftVec_(shiftVec), cj4_(cj4), sci_(sci), excl_(excl),
      atomTypes_(atomTypes), nbfp_(nbfp), numTypes_(numTypes),
      rCoulombSq_(rCoulombSq), ewaldBeta_(ewaldBeta), epsFac_(epsFac),
      deltaF_(NUM_ATOMS), deltaFShift_(c_numIvecs)
  {}

  // Computes the force / shift-force contribution of a SINGLE kernel
  // launch.
  void computeDelta(bool calcShift)
  {
    std::fill(deltaF_.begin(), deltaF_.end(), HostVec3{});
    std::fill(deltaFShift_.begin(), deltaFShift_.end(), HostVec3{});

    const float beta2 = ewaldBeta_ * ewaldBeta_;
    const float beta3 = ewaldBeta_ * ewaldBeta_ * ewaldBeta_;
    constexpr unsigned superClMask =
        ((1U << c_nbnxnGpuNumClusterPerSupercluster) - 1U);
    constexpr int prunedClusterPairSize = c_clSize * c_splitClSize;

    for (int bidx = 0; bidx < grid_z; ++bidx) {
      const nbnxn_sci_t nbSci = sci_[bidx];
      const int sciIdx    = nbSci.sci;
      const int cij4Start = nbSci.cj4_ind_start;
      const int cij4End   = nbSci.cj4_ind_end;
      const int shift     = nbSci.shift;
      const bool doCalcShift = (calcShift && shift != c_centralShiftIndex);

      for (int j4 = cij4Start; j4 < cij4End; ++j4) {
        for (int jm = 0; jm < c_nbnxnGpuJgroupSize; ++jm) {
          const int cj = cj4_[j4].cj[jm];

          for (int tidxj = 0; tidxj < c_clSize; ++tidxj) {
            const int aj = cj * c_clSize + tidxj;
            const int imeiIdx = tidxj / c_splitClSize;
            const unsigned imask = cj4_[j4].imei[imeiIdx].imask;

            // Fast skip: none of this jm's bits are set for this imeiIdx.
            if (!(imask & (superClMask << (jm * c_nbnxnGpuNumClusterPerSupercluster))))
              continue;

            const int wexclIdx = cj4_[j4].imei[imeiIdx].excl_ind;

            const Float4 xqjRaw = xq_[aj];
            const float xj = xqjRaw[0], yj = xqjRaw[1], zj = xqjRaw[2], qj = xqjRaw[3];

            for (int tidxi = 0; tidxi < c_clSize; ++tidxi) {
              const int tidx = tidxi + tidxj * c_clSize;
              const unsigned wexcl =
                  excl_[wexclIdx].pair[tidx & (prunedClusterPairSize - 1)];
              const bool nonSelf =
                  !(shift == c_centralShiftIndex && tidxj <= (unsigned)tidxi);

              for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; ++i) {
                const unsigned maskJI =
                    (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster + i));
                if (!(imask & maskJI)) continue;

                const int ci = sciIdx * c_nbnxnGpuNumClusterPerSupercluster + i;
                const int ai = ci * c_clSize + tidxi;

                const Float3 shiftV = shiftVec_[shift];
                const Float4 xqiRaw = xq_[ai];
                const float xi = xqiRaw[0] + shiftV[0];
                const float yi = xqiRaw[1] + shiftV[1];
                const float zi = xqiRaw[2] + shiftV[2];
                const float qi = xqiRaw[3] * epsFac_;

                const float rvx = xi - xj, rvy = yi - yj, rvz = zi - zj;
                float r2 = rvx * rvx + rvy * rvy + rvz * rvz;

                const float pairExclMask = (wexcl & maskJI) ? 1.0f : 0.0f;
                const bool notExcluded = nonSelf || (ci != cj);

                if (!(r2 < rCoulombSq_) || !notExcluded) continue;

                const int atomTypeI = atomTypes_[ai];
                const int atomTypeJ = atomTypes_[aj];
                const Float2 c6c12 = nbfp_[numTypes_ * atomTypeI + atomTypeJ];
                const float c6 = c6c12[0], c12 = c6c12[1];

                r2 = std::max(r2, c_nbnxnMinDistanceSquared);
                const float rInv  = 1.0f / std::sqrt(r2);
                const float r2Inv = rInv * rInv;
                float r6Inv = r2Inv * r2Inv * r2Inv;
                r6Inv *= pairExclMask;
                float fInvR = r6Inv * (c12 * r6Inv - c6) * r2Inv;
                fInvR += qi * qj *
                    (pairExclMask * r2Inv * rInv + pmeCorrF_host(beta2 * r2) * beta3);

                const float fx = rvx * fInvR, fy = rvy * fInvR, fz = rvz * fInvR;

                deltaF_[ai].x += fx; deltaF_[ai].y += fy; deltaF_[ai].z += fz;
                deltaF_[aj].x -= fx; deltaF_[aj].y -= fy; deltaF_[aj].z -= fz;

                if (doCalcShift) {
                  deltaFShift_[shift].x += fx;
                  deltaFShift_[shift].y += fy;
                  deltaFShift_[shift].z += fz;
                }
              }
            }
          }
        }
      }
    }
  }

  // Compares initValue + launchCount * delta against the GPU buffers using
  // a combined absolute+relative tolerance:
  //
  //     |got - expected| <= absTol + relTol * max(|got|, |expected|)
  bool validate(const Float3* gpu_f, const Float3* gpu_fShift,
                int launchCount,
                float initFx, float initFy, float initFz,
                float initFShiftX, float initFShiftY, float initFShiftZ,
                float absTol,
                const char* label,
                float relTol = 1e-3f,
                int maxReportedMismatches = 10) const
  {
    bool ok = true;
    int reported = 0;

    auto withinTol = [&](float got, float expected) {
      const float diff = std::fabs(got - expected);
      const float bound = absTol + relTol * std::max(std::fabs(got), std::fabs(expected));
      return diff <= bound;
    };

    for (int i = 0; i < NUM_ATOMS; ++i) {
      const float ex = initFx + launchCount * deltaF_[i].x;
      const float ey = initFy + launchCount * deltaF_[i].y;
      const float ez = initFz + launchCount * deltaF_[i].z;
      const Float3 gv = gpu_f[i];
      if (!withinTol(gv[0], ex) || !withinTol(gv[1], ey) || !withinTol(gv[2], ez)) {
        ok = false;
        if (reported < maxReportedMismatches) {
          printf("[%s] a_f[%d] mismatch: got (%f, %f, %f) expected (%f, %f, %f)\n",
                 label, i, gv[0], gv[1], gv[2], ex, ey, ez);
          ++reported;
        }
      }
    }

    for (int s = 0; s < c_numIvecs; ++s) {
      const float ex = initFShiftX + launchCount * deltaFShift_[s].x;
      const float ey = initFShiftY + launchCount * deltaFShift_[s].y;
      const float ez = initFShiftZ + launchCount * deltaFShift_[s].z;
      const Float3 gv = gpu_fShift[s];
      if (!withinTol(gv[0], ex) || !withinTol(gv[1], ey) || !withinTol(gv[2], ez)) {
        ok = false;
        if (reported < maxReportedMismatches) {
          printf("[%s] fShift[%d] mismatch: got (%f, %f, %f) expected (%f, %f, %f)\n",
                 label, s, gv[0], gv[1], gv[2], ex, ey, ez);
          ++reported;
        }
      }
    }

    printf("[%s] validation: %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
  }

private:
  const Float4*        xq_;
  const Float3*        shiftVec_;
  const nbnxn_cj4_t*   cj4_;
  const nbnxn_sci_t*   sci_;
  const nbnxn_excl_t*  excl_;
  const int*           atomTypes_;
  const Float2*        nbfp_;
  int   numTypes_;
  float rCoulombSq_;
  float ewaldBeta_;
  float epsFac_;

  std::vector<HostVec3> deltaF_;
  std::vector<HostVec3> deltaFShift_;
};
