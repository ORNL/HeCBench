// BLS12-381 G1 multi-scalar multiplication, the operation computed by
// sppark's mult_pippenger (poc/msm-cuda/cuda/pippenger.cu, mont=false).
//
// Scalars are 255-bit integers. The window width follows sppark's rule
// (10 to 18 bits, from the input length). Each scalar is recoded into
// signed radix-2^w digits that sum back to the integer, so [s]P matches
// a double-and-add of s. Points that share a digit are summed once.
// The running sum 1*B[1] + 2*B[2] + ... turns bucket k into k times its
// sum, then each window is shifted by w doublings and added to the next,
// which is the same window fold as sppark's collect().
// Derived from sppark (Supranational LLC), Apache-2.0.

#pragma once

#include <cstdint>

#if defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#define MSM_HD __host__ __device__
#else
#define MSM_HD
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__SYCL_DEVICE_ONLY__)
#define MSM_NOINLINE __attribute__((noinline))
#else
#define MSM_NOINLINE
#endif

static constexpr int kScalarBits = 255;
static constexpr int kBlock = 128;
// Smallest window is 10 bits, so a 255-bit scalar needs at most 26 digits.
static constexpr int kMaxWins = 26;

// -p^{-1} mod 2^64, the constant blst_384_t is instantiated with.
static constexpr uint64_t kN0 = 0x89f3fffcfffcfffdull;

// Limb accessors, not namespace arrays: nvcc does not emit a file-scope
// constexpr array into device code.
MSM_HD inline uint64_t prime_limb(int i) {
  switch (i) {
    case 0: return 0xb9feffffffffaaabull;
    case 1: return 0x1eabfffeb153ffffull;
    case 2: return 0x6730d2a0f6b0f624ull;
    case 3: return 0x64774b84f38512bfull;
    case 4: return 0x4b1ba7b6434bacd7ull;
    default: return 0x1a0111ea397fe69aull;
  }
}

struct wide_t {
  uint64_t lo, hi;
};

MSM_HD inline wide_t mul64(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return wide_t{a * b, __umul64hi(a, b)};
#elif defined(__SYCL_DEVICE_ONLY__)
  return wide_t{a * b, sycl::mul_hi(a, b)};
#else
  unsigned __int128 p = (unsigned __int128)a * b;
  return wide_t{(uint64_t)p, (uint64_t)(p >> 64)};
#endif
}

// a * b + c + d, which always fits in two words.
MSM_HD inline uint64_t mac(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t& hi) {
  wide_t p = mul64(a, b);
  uint64_t lo = p.lo + c;
  uint64_t h = p.hi + (lo < c ? 1ull : 0ull);
  uint64_t r = lo + d;
  hi = h + (r < d ? 1ull : 0ull);
  return r;
}

MSM_HD inline wide_t add3(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t s = a + b;
  uint64_t c1 = s < a ? 1ull : 0ull;
  uint64_t s2 = s + c;
  uint64_t c2 = s2 < s ? 1ull : 0ull;
  return wide_t{s2, c1 + c2};
}

// Field element in Montgomery form (value * 2^384 mod p), except the
// raw constructors used before to_mont().
struct fp {
  uint64_t v[6];
};

MSM_HD inline fp prime() {
  fp r;
  for (int i = 0; i < 6; i++) r.v[i] = prime_limb(i);
  return r;
}

MSM_HD inline fp fp_zero() {
  fp r;
  for (int i = 0; i < 6; i++) r.v[i] = 0;
  return r;
}

MSM_HD inline fp fp_from_u64(uint64_t x) {
  fp r = fp_zero();
  r.v[0] = x;
  return r;
}

MSM_HD inline bool fp_eq(fp a, fp b) {
  for (int i = 0; i < 6; i++)
    if (a.v[i] != b.v[i]) return false;
  return true;
}

MSM_HD inline bool fp_is0(fp a) { return fp_eq(a, fp_zero()); }

MSM_HD inline bool fp_ge_p(fp a) {
  for (int i = 5; i >= 0; --i) {
    if (a.v[i] > prime_limb(i)) return true;
    if (a.v[i] < prime_limb(i)) return false;
  }
  return true;
}

// Subtract p. Returns the borrow out of the top limb.
MSM_HD inline uint64_t fp_sub_p(fp& a) {
  uint64_t br = 0;
  for (int i = 0; i < 6; i++) {
    uint64_t ai = a.v[i];
    uint64_t pi = prime_limb(i);
    a.v[i] = ai - pi - br;
    br = (ai < pi) || (br && ai == pi) ? 1ull : 0ull;
  }
  return br;
}

MSM_HD inline fp fadd(fp a, fp b) {
  fp r;
  uint64_t c = 0;
  for (int i = 0; i < 6; i++) {
    wide_t s = add3(a.v[i], b.v[i], c);
    r.v[i] = s.lo;
    c = s.hi;
  }
  if (c || fp_ge_p(r)) fp_sub_p(r);
  return r;
}

MSM_HD inline fp fsub(fp a, fp b) {
  fp r;
  uint64_t br = 0;
  for (int i = 0; i < 6; i++) {
    uint64_t ai = a.v[i], bi = b.v[i];
    r.v[i] = ai - bi - br;
    br = (ai < bi) || (br && ai == bi) ? 1ull : 0ull;
  }
  if (br) {
    uint64_t c = 0;
    for (int i = 0; i < 6; i++) {
      wide_t s = add3(r.v[i], prime_limb(i), c);
      r.v[i] = s.lo;
      c = s.hi;
    }
  }
  return r;
}

MSM_HD inline fp fneg(fp a) {
  if (fp_is0(a)) return a;
  return fsub(fp_zero(), a);
}

// CIOS Montgomery multiply. Inputs and output are in Montgomery form
// and reduced mod p. The top limb of p is below 2^62, so the running
// value fits in six words and the result is below 2p.
MSM_HD inline MSM_NOINLINE fp fmul(fp a, fp b) {
  fp t = fp_zero();
  for (int i = 0; i < 6; i++) {
    uint64_t A, C;
    t.v[0] = mac(a.v[0], b.v[i], t.v[0], 0, A);
    uint64_t m = t.v[0] * kN0;
    mac(m, prime_limb(0), t.v[0], 0, C);
    for (int j = 1; j < 6; j++) {
      t.v[j] = mac(a.v[j], b.v[i], t.v[j], A, A);
      t.v[j - 1] = mac(m, prime_limb(j), t.v[j], C, C);
    }
    t.v[5] = C + A;
  }
  if (fp_ge_p(t)) fp_sub_p(t);
  return t;
}

MSM_HD inline fp fp_load(const uint64_t src[6]) {
  fp r;
  for (int i = 0; i < 6; i++) r.v[i] = src[i];
  return r;
}

MSM_HD inline fp to_mont(fp normal) {
  // 2^768 mod p.
  fp r2;
  r2.v[0] = 0xf4df1f341c341746ull;
  r2.v[1] = 0x0a76e6a609d104f1ull;
  r2.v[2] = 0x8de5476c4c95b6d5ull;
  r2.v[3] = 0x67eb88a9939d83c0ull;
  r2.v[4] = 0x9a793e85b519952dull;
  r2.v[5] = 0x11988fe592cae3aaull;
  return fmul(normal, r2);
}

MSM_HD inline fp from_mont(fp mont) {
  fp one = fp_from_u64(1);
  return fmul(mont, one);
}

MSM_HD inline fp mont_one() {
  // 2^384 mod p.
  fp r;
  r.v[0] = 0x760900000002fffdull;
  r.v[1] = 0xebf4000bc40c0002ull;
  r.v[2] = 0x5f48985753c758baull;
  r.v[3] = 0x77ce585370525745ull;
  r.v[4] = 0x5c071a97a256ec6dull;
  r.v[5] = 0x15f65ec3fa80e493ull;
  return r;
}

// G1 generator, normal form (not Montgomery). y^2 = x^3 + 4.
MSM_HD inline fp gen_x() {
  uint64_t x[6] = {0xfb3af00adb22c6bbull, 0x6c55e83ff97a1aefull, 0xa14e3a3f171bac58ull,
                   0xc3688c4f9774b905ull, 0x2695638c4fa9ac0full, 0x17f1d3a73197d794ull};
  return fp_load(x);
}
MSM_HD inline fp gen_y() {
  uint64_t y[6] = {0x0caa232946c5e7e1ull, 0xd03cc744a2888ae4ull, 0x00db18cb2c04b3edull,
                   0xfcf5e095d5d00af6ull, 0xa09e30ed741d8ae4ull, 0x08b3f481e3aaa0f1ull};
  return fp_load(y);
}

// Same rule as msm_t::msm_t in sppark's pippenger.cuh, including the
// round-up of the point count to a multiple of the warp size.
MSM_HD inline int choose_wbits(uint32_t npoints) {
  uint32_t n = (npoints + 31u) & ~31u;
  if (n > 192) {
    uint32_t m = n + n / 2;
    int lg = 0;
    while (m >>= 1) lg++;
    int w = lg - 8;
    if (w > 18) w = 18;
    if (w < 10) w = 10;
    return w;
  }
  return n ? 10 : 17;
}

MSM_HD inline int choose_nwins(int wbits) { return (kScalarBits - 1) / wbits + 1; }

// Largest absolute signed digit. A full-width top window keeps the carry,
// so its digit can be 2^wbits; every other digit fits in 2^(wbits-1).
MSM_HD inline int choose_max_digit(int wbits) {
  int last = kScalarBits % wbits;
  if (last == 0) return 1 << wbits;
  return 1 << (wbits - 1);
}

struct sched_t {
  int wbits, nwins, max_digit, nbins;
};

MSM_HD inline sched_t choose_sched(uint32_t npoints) {
  sched_t s;
  s.wbits = choose_wbits(npoints);
  s.nwins = choose_nwins(s.wbits);
  s.max_digit = choose_max_digit(s.wbits);
  s.nbins = s.max_digit + 1;
  return s;
}

struct scalar_t {
  uint64_t v[4];
};

// Signed windows. Reconstruct with
//   (...((d[n-1]) << w + d[n-2]) << w ... ) + d[0]  ==  the 255-bit scalar.
MSM_HD inline void scalar_windows(scalar_t s, int32_t* digits, int wbits, int nwins) {
  int carry = 0;
  int base = 1 << wbits;
  int half = base >> 1;
  int done = 0;
  for (int i = 0; i < nwins; i++) {
    int n = wbits;
    if (done >= kScalarBits) n = 0;
    else if (done + n > kScalarBits) n = kScalarBits - done;
    uint32_t bits = 0;
    if (n) {
      int limb = done >> 6;
      int sh = done & 63;
      uint64_t v = s.v[limb] >> sh;
      if (sh + n > 64 && limb + 1 < 4) v |= s.v[limb + 1] << (64 - sh);
      uint64_t mask = (n == 64) ? ~0ull : ((1ull << n) - 1ull);
      bits = (uint32_t)(v & mask);
      done += n;
    }
    bits += (uint32_t)carry;
    if (done < kScalarBits && (int)bits > half) {
      digits[i] = (int)bits - base;
      carry = 1;
    } else {
      digits[i] = (int)bits;
      carry = 0;
    }
  }
}

// Affine point in Montgomery form.
struct point_t {
  fp x, y;
};

struct jac_t {
  fp x, y, z;
};

MSM_HD inline jac_t jinf() {
  jac_t r;
  r.x = fp_zero();
  r.y = mont_one();
  r.z = fp_zero();
  return r;
}

MSM_HD inline bool j_is_inf(jac_t p) { return fp_is0(p.z); }

MSM_HD inline jac_t jfrom(point_t p) {
  jac_t r;
  r.x = p.x;
  r.y = p.y;
  r.z = mont_one();
  return r;
}

// Jacobian doubling for a = 0 (BLS12-381 G1).
MSM_HD inline jac_t jdbl(jac_t p) {
  if (j_is_inf(p) || fp_is0(p.y)) return jinf();
  fp A = fmul(p.x, p.x);
  fp B = fmul(p.y, p.y);
  fp C = fmul(B, B);
  fp xb = fadd(p.x, B);
  fp D = fsub(fmul(xb, xb), fadd(A, C));
  D = fadd(D, D);
  fp E = fadd(A, fadd(A, A));
  fp F = fmul(E, E);
  fp X3 = fsub(F, fadd(D, D));
  fp C8 = fadd(C, C);
  C8 = fadd(C8, C8);
  C8 = fadd(C8, C8);
  fp Y3 = fsub(fmul(E, fsub(D, X3)), C8);
  fp Z3 = fmul(fadd(p.y, p.y), p.z);
  jac_t r;
  r.x = X3;
  r.y = Y3;
  r.z = Z3;
  return r;
}

MSM_HD inline jac_t jadd(jac_t p, jac_t q) {
  if (j_is_inf(p)) return q;
  if (j_is_inf(q)) return p;
  fp Z1Z1 = fmul(p.z, p.z);
  fp Z2Z2 = fmul(q.z, q.z);
  fp U1 = fmul(p.x, Z2Z2);
  fp U2 = fmul(q.x, Z1Z1);
  fp S1 = fmul(fmul(p.y, q.z), Z2Z2);
  fp S2 = fmul(fmul(q.y, p.z), Z1Z1);
  fp H = fsub(U2, U1);
  fp r = fadd(fsub(S2, S1), fsub(S2, S1));
  if (fp_is0(H)) return fp_is0(r) ? jdbl(p) : jinf();
  fp HH = fmul(H, H);
  fp I = fadd(HH, HH);
  I = fadd(I, I);
  fp J = fmul(H, I);
  fp V = fmul(U1, I);
  fp X3 = fsub(fmul(r, r), fadd(J, fadd(V, V)));
  fp Y3 = fsub(fmul(r, fsub(V, X3)), fmul(fadd(S1, S1), J));
  fp zz = fmul(fadd(p.z, q.z), fadd(p.z, q.z));
  fp Z3 = fmul(fsub(zz, fadd(Z1Z1, Z2Z2)), H);
  jac_t o;
  o.x = X3;
  o.y = Y3;
  o.z = Z3;
  return o;
}

// Jacobian plus affine (madd-2007-bl).
MSM_HD inline jac_t jadd_mixed(jac_t p, point_t q) {
  if (j_is_inf(p)) return jfrom(q);
  fp Z1Z1 = fmul(p.z, p.z);
  fp U2 = fmul(q.x, Z1Z1);
  fp S2 = fmul(fmul(q.y, p.z), Z1Z1);
  fp H = fsub(U2, p.x);
  fp r = fadd(fsub(S2, p.y), fsub(S2, p.y));
  if (fp_is0(H)) return fp_is0(r) ? jdbl(p) : jinf();
  fp HH = fmul(H, H);
  fp I = fadd(HH, HH);
  I = fadd(I, I);
  fp J = fmul(H, I);
  fp V = fmul(p.x, I);
  fp X3 = fsub(fmul(r, r), fadd(J, fadd(V, V)));
  fp Y3 = fsub(fmul(r, fsub(V, X3)), fmul(fadd(p.y, p.y), J));
  fp zh = fadd(p.z, H);
  fp Z3 = fsub(fmul(zh, zh), fadd(Z1Z1, HH));
  jac_t o;
  o.x = X3;
  o.y = Y3;
  o.z = Z3;
  return o;
}

MSM_HD inline jac_t jmul(jac_t p, uint32_t k) {
  jac_t r = jinf();
  for (int b = 31; b >= 0; --b) {
    r = jdbl(r);
    if ((k >> b) & 1u) r = jadd(r, p);
  }
  return r;
}

MSM_HD inline jac_t fold_windows(const jac_t* wins, int nwins, int wbits) {
  jac_t r = wins[nwins - 1];
  for (int w = nwins - 2; w >= 0; --w) {
    for (int i = 0; i < wbits; i++) r = jdbl(r);
    r = jadd(r, wins[w]);
  }
  return r;
}
