#ifndef NTT_REFERENCE_H
#define NTT_REFERENCE_H

#include <vector>

#include "modP.h"

inline void reference(uint32 *__restrict dst,
                      const uint64 *__restrict src,
                      int nttLen,
                      int blockSize,
                      int threadsPerBlock) {
  std::vector<uint64> buffer(blockSize);

  for (int block = 0; block < nttLen / blockSize; block++) {
    for (int thread = 0; thread < threadsPerBlock; thread++) {
      uint64 samples[8];
      const uint32 fmem =
          (block << 9) | ((thread & 0x3E) << 3) | (thread & 0x1);
      const uint32 tbuf = thread << 3;

      for (int i = 0; i < 8; i++)
        samples[i] = src[fmem | (i << 1)];
      ntt8(samples);

      for (int i = 0; i < 8; i++)
        buffer[tbuf | i] =
            _ls_modP(samples[i], ((thread & 0x1) << 2) * i * 3);
    }

    for (int thread = 0; thread < threadsPerBlock; thread++) {
      uint64 samples[8], s8[8];
      const uint32 fbuf = ((thread & 0x38) << 3) | (thread & 0x7);
      const uint32 tmem =
          (block << 9) | ((thread & 0x38) << 3) | (thread & 0x7);

      for (int i = 0; i < 8; i++)
        samples[i] = buffer[fbuf | (i << 3)];

      for (int i = 0; i < 4; i++) {
        s8[2 * i] = _add_modP(samples[2 * i], samples[2 * i + 1]);
        s8[2 * i + 1] = _sub_modP(samples[2 * i], samples[2 * i + 1]);
      }

      for (int i = 0; i < 8; i++) {
        const uint32 output =
            (((tmem | (i << 3)) & 0xf) << 12) |
            ((tmem | (i << 3)) >> 4);
        dst[output] =
            (uint32)_mul_modP(s8[i], 18446462594437939201UL, valP);
      }
    }
  }
}

#endif
