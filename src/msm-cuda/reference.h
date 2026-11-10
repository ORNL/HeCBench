// Host reference for the BLS12-381 G1 Pippenger MSM.
// pippenger_ref reads unsigned c-bit windows straight from the scalar
// bits, with c independent of the device window, so it shares neither
// the signed recoding nor the bucket schedule. The affine double-and-add
// checks it, the group law, and the windows on a small instance.

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

inline bool jac_equal(jac_t a, jac_t b) { return aff_equal(jac_to_aff(a), b); }

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

inline bool on_curve(point_t p) {
  fp four = to_mont(fp_from_u64(4));
  fp y2 = fmul(p.y, p.y);
  fp x3 = fmul(fmul(p.x, p.x), p.x);
  return fp_eq(y2, fadd(x3, four));
}

// P_i = (i+1) * G, scalars uniform in [0, 2^255). The Jacobian z of
// every point is inverted at once with Montgomery's trick.
inline void make_instance(size_t n, uint64_t seed, point_t* points, scalar_t* scalars) {
  point_t g{to_mont(gen_x()), to_mont(gen_y())};
  std::vector<jac_t> jp(n);
  std::vector<fp> prefix(n);
  jac_t p = jfrom(g);
  fp run = mont_one();
  for (size_t i = 0; i < n; i++) {
    jp[i] = p;
    prefix[i] = run;
    run = fmul(run, p.z);
    p = jadd_mixed(p, g);
  }
  fp inv = to_mont(ninv(from_mont(run)));
  for (size_t i = n; i-- > 0;) {
    fp iz = fmul(inv, prefix[i]);
    inv = fmul(inv, jp[i].z);
    fp iz2 = fmul(iz, iz);
    points[i].x = fmul(jp[i].x, iz2);
    points[i].y = fmul(fmul(jp[i].y, iz), iz2);
  }
  uint64_t x = seed;
  for (size_t i = 0; i < n; i++) {
    for (int k = 0; k < 4; k++) {
      x += 0x9e3779b97f4a7c15ull;
      uint64_t z = x;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
      scalars[i].v[k] = z ^ (z >> 31);
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

inline uint32_t scalar_bits(const scalar_t& s, int pos, int c) {
  uint32_t d = 0;
  for (int b = 0; b < c && pos + b < kScalarBits; b++)
    d |= (uint32_t)((s.v[(pos + b) >> 6] >> ((pos + b) & 63)) & 1ull) << b;
  return d;
}

inline jac_t pippenger_ref(const point_t* points, const scalar_t* scalars, size_t n) {
  int lg = 0;
  while (((size_t)2 << lg) <= n) lg++;
  int c = lg - 3;
  if (c < 4) c = 4;
  if (c > 16) c = 16;
  const int nwins = (kScalarBits + c - 1) / c;
  std::vector<jac_t> buckets((size_t)1 << c);
  jac_t r = jinf();
  for (int w = nwins - 1; w >= 0; --w) {
    for (int i = 0; i < c; i++) r = jdbl(r);
    for (auto& b : buckets) b = jinf();
    for (size_t i = 0; i < n; i++) {
      uint32_t d = scalar_bits(scalars[i], w * c, c);
      if (d) buckets[d] = jadd_mixed(buckets[d], points[i]);
    }
    jac_t acc = jinf();
    jac_t sum = jinf();
    for (size_t k = buckets.size() - 1; k >= 1; --k) {
      acc = jadd(acc, buckets[k]);
      sum = jadd(sum, acc);
    }
    r = jadd(r, sum);
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
  if (!fp_eq(nmul(pm1, pm1), one)) return false;

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

  if (choose_wbits(32) != 10 || choose_wbits(1u << 18) != 10 || choose_wbits(1u << 20) != 12 ||
      choose_wbits(1u << 23) != 15)
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
  aff_t three_g = aadd(two_g, g);
  point_t gm{to_mont(gen_x()), to_mont(gen_y())};
  jac_t J = jfrom(gm);
  if (!aff_equal(two_g, jdbl(J))) return false;
  if (!aff_equal(two_g, jadd(J, J))) return false;
  if (!aff_equal(two_g, jadd_mixed(J, gm))) return false;
  if (!aff_equal(three_g, jadd_mixed(jdbl(J), gm))) return false;
  if (!aff_equal(three_g, jmul(J, 3))) return false;
  point_t neg = gm;
  neg.y = fneg(neg.y);
  if (!j_is_inf(jadd(J, jfrom(neg)))) return false;
  if (!j_is_inf(jadd_mixed(J, neg))) return false;

  const size_t n = 6;
  point_t points[n];
  scalar_t scalars[n];
  make_instance(n, 0x9e37, points, scalars);
  for (size_t i = 0; i < n; i++)
    if (!on_curve(points[i])) return false;
  if (!aff_equal(three_g, jfrom(points[2]))) return false;
  scalars[0] = edges[0];
  scalars[1] = edges[1];
  scalars[2] = edges[2];
  scalars[3] = edges[3];
  aff_t naive = reference_msm(points, scalars, n);
  return aff_equal(naive, pippenger_ref(points, scalars, n));
}
