#if (defined(__CUDACC__) || defined(__HIPCC__))
  #define ESS __inline__ __host__ __device__
#else
  #define ESS inline
#endif

#define valP 0xffffffff00000001UL // special prime number for NTT conversions
#define uint32Max 0xffffffffU     // maximum value of an unsigned integer

typedef      unsigned int uint32; // 32-bit unsigned integer
typedef unsigned long int uint64; // 64-bit unsigned integer

#ifdef _OPENMP
#pragma omp declare target
#endif
ESS
void _uint96_modP (uint32 *x) {
  uint64 s = (uint64)x[1] + (uint64)x[2];
  int b = (s > 4294967295UL) ? 1 : 0;
  x[1] = (uint32) s; 
  if (x[0] < x[2]) {
    if (x[1] < 1) {
      x[1]--;
      b--;
    }
  }
  x[0] -= x[2];
  if (b == 1) {
    s = (uint64)x[0] + 4294967295UL;
    if (s > 4294967295UL) x[1]++;
    x[0] = (uint32) s; 
  }
}

ESS
void _uint128_modP (uint32 *x) {
  _uint96_modP (x);
  int b = 0;
  if (x[0] < x[3])  {
    if (x[1] < 1) {
      x[1]--;
      b = -1;
    }
  }
  x[0] -= x[3];
  if (b == -1) {
    if (x[0] < 4294967295U) x[1]--;
    x[0] -= 4294967295U;
  }
}

ESS
void _uint160_modP (uint32 *x) {
  _uint128_modP (x);
  int b = (x[1] < x[4]) ? -1 : 0;
  x[1] -= x[4];
  if (b == -1) {
    if (x[0] < 4294967295U) x[1]--;
    x[0] -= 4294967295U;
  }
}

ESS
void _uint192_modP (uint32 *x) {
  _uint160_modP (x);
  uint64 s = (uint64)x[0] + (uint64)x[5];
  x[0] = (uint32) s; 
  int c = (s > 4294967295UL) ? 1 : 0;
  s = (uint64)x[1] + (uint64)c;
  int b = (s > 4294967295UL) ? 1 : 0;
  x[1] = (uint32)s;
  if (x[1] < x[5]) b--;
  x[1] -= x[5];
  if (b == -1) {
    if (x[0] < 4294967295U) x[1]--;
    x[0] -= 4294967295U;
  }
}

ESS
void _uint224_modP (uint32 *x) {
  _uint192_modP (x);
  uint64 s = (uint64)x[0] + (uint64)x[6];
  int c = (s > 4294967295UL) ? 1 : 0;
  x[0] = (uint32) s; 
  s = (uint64)x[1] + (uint64)c;
  x[1] = (uint32) s;
  c = (s > 4294967295UL) ? 1 : 0;
  if (c == 1) {
    s = (uint64)x[0] + 4294967295UL; 
    if (s > 4294967295UL) x[1]++;
    x[0] = (uint32) s;
  }
}

ESS
uint64 _ls_modP(uint64 x, int l) {
  uint64 tx = x;
  uint32 buff[7];
  switch(l){
    // 2 words
    case (0):
      buff[0] = (uint32)tx;
      buff[1] = (uint32)(tx>>32);
      break;
      // 3 words
    case (3):
    case (6):
    case (9):
    case (12):
    case (15):
    case (18):
    case (21):
    case (24):
    case (27):
    case (30):
      buff[2] = (uint32)(tx>>(64-l));
      buff[1] = (uint32)(tx>>(32-l));
      buff[0] = (uint32)(tx<<l);
      _uint96_modP(buff);
      break;
      // 4 words
    case (36):
    case (42):
    case (45):
    case (48):
    case (54):
    case (60):
    case (63):
      buff[3] = (uint32)(tx>>(96-l));
      buff[2] = (uint32)(tx>>(64-l));
      buff[1] = (uint32)(tx<<(l-32));
      buff[0] = 0;
      _uint128_modP(buff);
      break;
      // 5 words
    case (72):
    case (75):
    case (84):
    case (90):
      buff[4] = (uint32)(tx>>(128-l));
      buff[3] = (uint32)(tx>>(96-l));
      buff[2] = (uint32)(tx<<(l-64));
      buff[1] = 0;
      buff[0] = 0;
      _uint160_modP(buff);
      break;
      // 6 words
    case (105):
    case (108):
    case (126):
      buff[5] = (uint32)(tx>>(160-l));
      buff[4] = (uint32)(tx>>(128-l));
      buff[3] = (uint32)(tx<<(l-96));
      buff[2] = 0;
      buff[1] = 0;
      buff[0] = 0;
      _uint192_modP(buff);
      break;
      // 7 words
    case (147):
      buff[6] = (uint32)(tx>>(192-l));
      buff[5] = (uint32)(tx>>(160-l));
      buff[4] = (uint32)(tx<<(l-128));
      buff[3] = 0;
      buff[2] = 0;
      buff[1] = 0;
      buff[0] = 0;
      _uint224_modP(buff);
      break;
  }
  if (*(uint64 *)buff > valP)
    *(uint64 *)buff -= valP;
  return *(uint64 *)buff;
}

ESS
uint64 _add_modP(uint64 x, uint64 y) {
  uint64 ret;
  ret = x+y;
  if (ret < x) ret += uint32Max;
  if (ret >= valP) ret -= valP;
  return ret;
}

ESS
uint64 _sub_modP(uint64 x, uint64 y) {
   uint64 ret;
  ret = x-y;
  if (ret > x) ret -= uint32Max;
  return ret;
}

// reference: https://www.interdb.jp/blog/pgsql/modmul/
ESS
uint64 _mul_modP(uint64 x, uint64 y, uint64 m) {
  int i, bits;
  uint64 r = 0;

  /* Because of (x * y) % m = (x % m * y % m) % m */
  if (x >= m)
    x %= m;
  if (y >= m)
    y %= m;

  /* Return the trivial result. */
  if (x == 0 || y == 0 || m == 1)
    return 0;

  /* Return the result if (x * y) can be multiplicated without overflow. */
  if ((x | y) < (0xffffffffUL))
    return (x * y) % m;

  /* To reduce the for loop in the algorithm below. */
  if (x < y)
  {
    uint64 tmp = x;
    x = y;
    y = tmp;
  }

  /* Interleaved modular multiplication algorithm
   *
   *   D.N. Amanor, et al, "Efficient hardware architecture for
   *    modular multiplication on FPGAs", in Field Programmable
   *    Logic and Apllications, 2005. International Conference on,
   *    Aug 2005, pp. 539-542.
   *
   * This algorithm is usually used in the field of digital circuit
   * design.
   *
   * Input: X, Y, M; 0 <= X, Y <= M;
   * Output: R = X *  Y mod M;
   * bits: number of bits of Y
   * Y[i]: i th bit of Y
   *
   * 1. R = 0;
   * 2. for (i = bits - 1; i >= 0; i--) {
   * 3.   R = 2 * R;
   * 4.   if (Y[i] == 0x1)
   * 5.     R += X;
   * 6.   if (R >= M) R -= M;
   * 7.  if (R >= M) R -= M;
   *   }
   *
   * In Steps 3 and 5, overflow should be avoided.
   * Steps 6 and 7 can be instead of a modular operation (R %= M).
   */

  bits = 64;
  for (i = bits - 1; i >= 0; i--)
  {
    if (r > 0x7fffffffffffffffUL)
      /* To avoid overflow, transform from (2 * r) to
       * (2 * r) % m, and further transform to
       * mathematically equivalent form shown below:
       */
      r = m - ((m - r) << 1);
    else
      r <<= 1;

    if ((y >> i) & 0x1)
    {
      if (r > 0xffffffffffffffffUL - x)
        /* To calculate (r + x) without overflow, transform to (r + x) % m,
         * and transform to mathematically equivalent form (r + x - m).
         */
        r += x - m;
      else
        r += x;
    }
    r %= m;
  }
  return r;
}

// 8-point NTT
ESS
void ntt8(uint64 *x) {
  uint64 s[8], temp;
  s[0] = _add_modP(x[0], x[4]);
  s[1] = _sub_modP(x[0], x[4]);
  s[2] = _add_modP(x[2], x[6]);
  s[3] = _sub_modP(x[2], x[6]);
  s[4] = _add_modP(x[1], x[5]);
  s[5] = _sub_modP(x[1], x[5]);
  s[6] = _add_modP(x[3], x[7]);
  s[7] = _sub_modP(x[3], x[7]);
  x[0] = _add_modP(s[0], s[2]);
  x[2] = _sub_modP(s[0], s[2]);
  temp = _ls_modP(s[3], 48);
  x[1] = _add_modP(s[1], temp);
  x[3] = _sub_modP(s[1], temp);
  x[4] = _add_modP(s[4], s[6]);
  x[6] = _sub_modP(s[4], s[6]);
  temp = _ls_modP(s[7], 48);
  x[5] = _add_modP(s[5], temp);
  x[7] = _sub_modP(s[5], temp);
  s[0] = _add_modP(x[0], x[4]);
  s[4] = _sub_modP(x[0], x[4]);
  temp = _ls_modP(x[5], 24);
  s[1] = _add_modP(x[1], temp);
  s[5] = _sub_modP(x[1], temp);
  temp = _ls_modP(x[6], 48);
  s[2] = _add_modP(x[2], temp);
  s[6] = _sub_modP(x[2], temp);
  temp = _ls_modP(x[7], 72);
  s[3] = _add_modP(x[3], temp);
  s[7] = _sub_modP(x[3], temp);
  x[0] = s[0];
  x[1] = s[1];
  x[2] = s[2];
  x[3] = s[3];
  x[4] = s[4];
  x[5] = s[5];
  x[6] = s[6];
  x[7] = s[7];
}
#ifdef _OPENMP
#pragma omp end declare target
#endif
