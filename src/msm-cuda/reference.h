// Host reference for the BLS12-381 G1 Pippenger MSM.
// The affine double-and-add does not share the bucket schedule, so
// agreement checks the windows and the group law. pippenger_cpu walks
// the same buckets as the device, one point at a time.

#pragma once

#include <cstdio>
#include <vector>

#include "msm.h"

struct aff_t {
  fp x, y;
  bool inf;
};

inline aff_t aff_inf() { return aff_t{fp_zero(), fp_zero(), true}; }

inline fp nmul(fp a, fp b) { return from_mont(fmul(to_mont(a), to_mont(b))); }

// a^(p-2) mod p, with the exponentiation done in Montgomery form.
inline fp ninv(fp a) {
  uint64_t e[6] = {prime_limb(0) - 2, prime_limb(1), prime_limb(2),
                   prime_limb(3), prime_limb(4), prime_limb(5)};
  fp base = to_mont(a);
  fp r = mont_one();
  for (int i = 5; i >= 0; --i) {
    for (int b = 63; b >= 0; --b) {
      r = fmul(r, r);
      if ((e[i] >> b) & 1ull) r = fmul(r, base);
    }
  }
  return from_mont(r);
}

inline aff_t jac_to_aff(jac_t p) {
  if (j_is_inf(p)) return aff_inf();
  fp iz = ninv(from_mont(p.z));
  fp iz2 = nmul(iz, iz);
  aff_t a;
  a.x = nmul(from_mont(p.x), iz2);
  a.y = nmul(nmul(from_mont(p.y), iz), iz2);
  a.inf = false;
  return a;
}

inline bool aff_equal(aff_t a, jac_t j) {
  aff_t b = jac_to_aff(j);
  if (a.inf || b.inf) return a.inf && b.inf;
  return fp_eq(a.x, b.x) && fp_eq(a.y, b.y);
}

inline void print_aff(const char* tag, aff_t a) {
  printf("%s inf=%d", tag, a.inf ? 1 : 0);
  if (!a.inf) {
    printf(" x");
    for (int i = 0; i < 6; i++) printf(" %016llx", (unsigned long long)a.x.v[i]);
    printf(" y");
    for (int i = 0; i < 6; i++) printf(" %016llx", (unsigned long long)a.y.v[i]);
  }
  printf("\n");
}

inline aff_t aadd(aff_t p, aff_t q) {
  if (p.inf) return q;
  if (q.inf) return p;
  if (fp_eq(p.x, q.x)) {
    if (!fp_eq(p.y, q.y) || fp_is0(p.y)) return aff_inf();
    fp lam = nmul(nmul(fp_from_u64(3), nmul(p.x, p.x)), ninv(fadd(p.y, p.y)));
    fp x3 = fsub(nmul(lam, lam), fadd(p.x, p.x));
    fp y3 = fsub(nmul(lam, fsub(p.x, x3)), p.y);
    return aff_t{x3, y3, false};
  }
  fp lam = nmul(fsub(q.y, p.y), ninv(fsub(q.x, p.x)));
  fp x3 = fsub(fsub(nmul(lam, lam), p.x), q.x);
  fp y3 = fsub(nmul(lam, fsub(p.x, x3)), p.y);
  return aff_t{x3, y3, false};
}

inline aff_t amul(aff_t p, scalar_t k) {
  aff_t r = aff_inf();
  for (int bit = kScalarBits - 1; bit >= 0; --bit) {
    if (!r.inf) r = aadd(r, r);
    if ((k.v[bit >> 6] >> (bit & 63)) & 1ull) r = aadd(r, p);
  }
  return r;
}

inline point_t jac_to_point(jac_t p) {
  fp iz = to_mont(ninv(from_mont(p.z)));
  fp iz2 = fmul(iz, iz);
  point_t o;
  o.x = fmul(p.x, iz2);
  o.y = fmul(fmul(p.y, iz), iz2);
  return o;
}

// P_i = (i+1) * G, scalars uniform in [0, 2^255).
inline void make_instance(size_t n, uint64_t seed, point_t* points, scalar_t* scalars) {
  point_t g{to_mont(gen_x()), to_mont(gen_y())};
  jac_t G = jfrom(g);
  jac_t p = G;
  for (size_t i = 0; i < n; i++) {
    points[i] = jac_to_point(p);
    p = jadd(p, G);
  }
  uint64_t x = seed ? seed : 1;
  for (size_t i = 0; i < n; i++) {
    for (int k = 0; k < 4; k++) {
      x = x * 6364136223846793005ull + 1;
      scalars[i].v[k] = x;
    }
    scalars[i].v[3] &= 0x7fffffffffffffffull;
  }
}

inline aff_t reference_msm(const point_t* points, const scalar_t* scalars, size_t n) {
  aff_t sum = aff_inf();
  for (size_t i = 0; i < n; i++) {
    aff_t p{from_mont(points[i].x), from_mont(points[i].y), false};
    sum = aadd(sum, amul(p, scalars[i]));
  }
  return sum;
}

inline jac_t pippenger_cpu(const point_t* points, const scalar_t* scalars, size_t n) {
  sched_t s = choose_sched((uint32_t)n);
  std::vector<jac_t> buckets((size_t)s.nwins * s.nbins, jinf());
  std::vector<int32_t> digits(s.nwins);
  for (size_t i = 0; i < n; i++) {
    scalar_windows(scalars[i], digits.data(), s.wbits, s.nwins);
    for (int w = 0; w < s.nwins; w++) {
      int dig = digits[w];
      if (dig == 0) continue;
      point_t p = points[i];
      int k = dig;
      if (k < 0) {
        p.y = fneg(p.y);
        k = -k;
      }
      size_t slot = (size_t)w * s.nbins + (size_t)k;
      buckets[slot] = jadd(buckets[slot], jfrom(p));
    }
  }
  std::vector<jac_t> wins(s.nwins);
  for (int w = 0; w < s.nwins; w++) {
    jac_t acc = jinf();
    jac_t sum = jinf();
    for (int k = s.max_digit; k >= 1; --k) {
      acc = jadd(acc, buckets[(size_t)w * s.nbins + (size_t)k]);
      sum = jadd(sum, acc);
    }
    wins[w] = sum;
  }
  jac_t r = wins[s.nwins - 1];
  for (int w = s.nwins - 2; w >= 0; --w) {
    for (int i = 0; i < s.wbits; i++) r = jdbl(r);
    r = jadd(r, wins[w]);
  }
  return r;
}

inline bool windows_reconstruct(scalar_t s, int wbits) {
  int nw = choose_nwins(wbits);
  int md = choose_max_digit(wbits);
  int32_t d[kMaxWins];
  scalar_windows(s, d, wbits, nw);
  uint64_t acc[5] = {0, 0, 0, 0, 0};
  for (int w = nw - 1; w >= 0; --w) {
    if (d[w] > md || d[w] < -md) return false;
    uint64_t carry = 0;
    for (int i = 0; i < 5; i++) {
      uint64_t cur = acc[i];
      acc[i] = (cur << wbits) | carry;
      carry = cur >> (64 - wbits);
    }
    if (carry) return false;
    if (d[w] >= 0) {
      uint64_t c = (uint32_t)d[w];
      for (int i = 0; c && i < 5; i++) {
        uint64_t sum = acc[i] + c;
        c = sum < acc[i] ? 1ull : 0ull;
        acc[i] = sum;
      }
      if (c) return false;
    } else {
      uint64_t br = (uint32_t)(-d[w]);
      for (int i = 0; i < 5; i++) {
        uint64_t ai = acc[i];
        acc[i] = ai - br;
        br = ai < br ? 1ull : 0ull;
        if (!br) break;
      }
      if (br) return false;
    }
  }
  for (int i = 0; i < 4; i++)
    if (acc[i] != s.v[i]) return false;
  return acc[4] == 0;
}

inline bool msm_self_test() {
  wide_t n0p = mul64(prime_limb(0), kN0);
  if (n0p.lo != ~uint64_t{0}) return false;

  fp one = fp_from_u64(1);
  fp two = fp_from_u64(2);
  if (!fp_eq(from_mont(to_mont(one)), one)) return false;
  if (!fp_eq(from_mont(to_mont(gen_x())), gen_x())) return false;
  if (!fp_eq(from_mont(to_mont(gen_y())), gen_y())) return false;

  fp pm1 = prime();
  pm1.v[0] -= 1;
  if (!fp_eq(from_mont(to_mont(pm1)), pm1)) return false;
  if (!fp_eq(from_mont(fadd(to_mont(pm1), to_mont(two))), one)) return false;

  fp a = fp_from_u64(0x123456789abcdefull);
  fp b = fp_from_u64(0xfedcba987654321ull);
  fp prod = from_mont(fmul(to_mont(a), to_mont(b)));
  if (prod.v[0] != 0x22236d88fe5618cfull || prod.v[1] != 0x00121fa00ad77d74ull) return false;
  for (int i = 2; i < 6; i++)
    if (prod.v[i] != 0) return false;
  fp sum = from_mont(fadd(to_mont(a), to_mont(b)));
  if (sum.v[0] != 0x1111111111111110ull) return false;
  for (int i = 1; i < 6; i++)
    if (sum.v[i] != 0) return false;

  fp y2 = nmul(gen_y(), gen_y());
  fp x3 = nmul(nmul(gen_x(), gen_x()), gen_x());
  if (!fp_eq(y2, fadd(x3, fp_from_u64(4)))) return false;

  if (choose_wbits(32) != 10 || choose_wbits(1u << 20) != 12 || choose_wbits(1u << 23) != 15)
    return false;

  scalar_t edges[4];
  for (int i = 0; i < 4; i++) edges[0].v[i] = 0;
  edges[1] = edges[0];
  edges[1].v[0] = 1;
  for (int i = 0; i < 3; i++) edges[2].v[i] = ~uint64_t{0};
  edges[2].v[3] = 0x7fffffffffffffffull;
  edges[3] = edges[0];
  edges[3].v[0] = 1023;
  const int widths[] = {10, 12, 15, 17, 18};
  uint64_t rng = 0x1234;
  for (int w : widths) {
    for (int e = 0; e < 4; e++)
      if (!windows_reconstruct(edges[e], w)) return false;
    for (int n = 0; n < 16; n++) {
      scalar_t s;
      for (int k = 0; k < 4; k++) {
        rng = rng * 6364136223846793005ull + 1;
        s.v[k] = rng;
      }
      s.v[3] &= 0x7fffffffffffffffull;
      if (!windows_reconstruct(s, w)) return false;
    }
  }

  aff_t g{gen_x(), gen_y(), false};
  aff_t two_g = aadd(g, g);
  point_t gm{to_mont(gen_x()), to_mont(gen_y())};
  jac_t J = jfrom(gm);
  if (!aff_equal(two_g, jdbl(J))) return false;
  if (!aff_equal(two_g, jadd(J, J))) return false;
  point_t neg = gm;
  neg.y = fneg(neg.y);
  if (!j_is_inf(jadd(J, jfrom(neg)))) return false;

  const size_t n = 4;
  point_t points[4];
  scalar_t scalars[4];
  make_instance(n, 0x9e37, points, scalars);
  scalars[0] = edges[0];
  scalars[1] = edges[1];
  scalars[2] = edges[2];
  aff_t naive = reference_msm(points, scalars, n);
  jac_t pip = pippenger_cpu(points, scalars, n);
  return aff_equal(naive, pip);
}
