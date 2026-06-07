/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 */
#include "common.h"
#include "flowGold.h"

inline float Tex2D(const float *t, int w, int h, int s, float x, float y) {
  float intPartX, intPartY;
  float dx = fabsf(modff(x, &intPartX));
  float dy = fabsf(modff(y, &intPartY));

  int ix0 = (int)intPartX;
  int iy0 = (int)intPartY;

  if (ix0 < 0) ix0 = abs(ix0 + 1);
  if (iy0 < 0) iy0 = abs(iy0 + 1);
  if (ix0 >= w) ix0 = w * 2 - ix0 - 1;
  if (iy0 >= h) iy0 = h * 2 - iy0 - 1;

  int ix1 = ix0 + 1;
  int iy1 = iy0 + 1;

  if (ix1 >= w) ix1 = w * 2 - ix1 - 1;
  if (iy1 >= h) iy1 = h * 2 - iy1 - 1;

  float res = t[ix0 + iy0 * s] * (1.0f - dx) * (1.0f - dy);
  res += t[ix1 + iy0 * s] * dx * (1.0f - dy);
  res += t[ix0 + iy1 * s] * (1.0f - dx) * dy;
  res += t[ix1 + iy1 * s] * dx * dy;

  return res;
}

inline float Tex2Di(const float *src, int w, int h, int s, int x, int y) {
  if (x < 0) x = abs(x + 1);
  if (y < 0) y = abs(y + 1);
  if (x >= w) x = w * 2 - x - 1;
  if (y >= h) y = h * 2 - y - 1;

  return src[x + y * s];
}

static void Downscale(const float *src, int width, int height, int stride,
                      int newWidth, int newHeight, int newStride, float *out) {
  for (int i = 0; i < newHeight; ++i) {
    for (int j = 0; j < newWidth; ++j) {
      const int srcX = j * 2;
      const int srcY = i * 2;
      float sum;
      sum = Tex2Di(src, width, height, stride, srcX + 0, srcY + 0);
      sum += Tex2Di(src, width, height, stride, srcX + 0, srcY + 1);
      sum += Tex2Di(src, width, height, stride, srcX + 1, srcY + 0);
      sum += Tex2Di(src, width, height, stride, srcX + 1, srcY + 1);
      sum *= 0.25f;
      out[j + i * newStride] = sum;
    }
  }
}

static void Upscale(const float *src, int width, int height, int stride,
                    int newWidth, int newHeight, int newStride, float scale,
                    float *out) {
  for (int i = 0; i < newHeight; ++i) {
    for (int j = 0; j < newWidth; ++j) {
      float x = ((float)j - 0.5f) * 0.5f;
      float y = ((float)i - 0.5f) * 0.5f;
      out[j + i * newStride] = Tex2D(src, width, height, stride, x, y) * scale;
    }
  }
}

static void WarpImage(const float *src, int w, int h, int s, const float *u,
                      const float *v, float *out) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int pos = j + i * s;
      float x = (float)j + u[pos];
      float y = (float)i + v[pos];
      out[pos] = Tex2D(src, w, h, s, x, y);
    }
  }
}

static void ComputeDerivatives(const float *I0, const float *I1, int w, int h,
                               int s, float *Ix, float *Iy, float *Iz) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int pos = j + i * s;
      float t0, t1;

      t0 = Tex2Di(I0, w, h, s, j - 2, i);
      t0 -= Tex2Di(I0, w, h, s, j - 1, i) * 8.0f;
      t0 += Tex2Di(I0, w, h, s, j + 1, i) * 8.0f;
      t0 -= Tex2Di(I0, w, h, s, j + 2, i);
      t0 /= 12.0f;

      t1 = Tex2Di(I1, w, h, s, j - 2, i);
      t1 -= Tex2Di(I1, w, h, s, j - 1, i) * 8.0f;
      t1 += Tex2Di(I1, w, h, s, j + 1, i) * 8.0f;
      t1 -= Tex2Di(I1, w, h, s, j + 2, i);
      t1 /= 12.0f;

      Ix[pos] = (t0 + t1) * 0.5f;
      Iz[pos] = I1[pos] - I0[pos];

      t0 = Tex2Di(I0, w, h, s, j, i - 2);
      t0 -= Tex2Di(I0, w, h, s, j, i - 1) * 8.0f;
      t0 += Tex2Di(I0, w, h, s, j, i + 1) * 8.0f;
      t0 -= Tex2Di(I0, w, h, s, j, i + 2);
      t0 /= 12.0f;

      t1 = Tex2Di(I1, w, h, s, j, i - 2);
      t1 -= Tex2Di(I1, w, h, s, j, i - 1) * 8.0f;
      t1 += Tex2Di(I1, w, h, s, j, i + 1) * 8.0f;
      t1 -= Tex2Di(I1, w, h, s, j, i + 2);
      t1 /= 12.0f;

      Iy[pos] = (t0 + t1) * 0.5f;
    }
  }
}

static void SolveForUpdate(const float *du0, const float *dv0, const float *Ix,
                           const float *Iy, const float *Iz, int w, int h,
                           int s, float alpha, float *du1, float *dv1) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int pos = j + i * s;
      int left, right, up, down;

      if (j != 0)
        left = pos - 1;
      else
        left = pos;

      if (j != w - 1)
        right = pos + 1;
      else
        right = pos;

      if (i != 0)
        down = pos - s;
      else
        down = pos;

      if (i != h - 1)
        up = pos + s;
      else
        up = pos;

      float sumU = (du0[left] + du0[right] + du0[up] + du0[down]) * 0.25f;
      float sumV = (dv0[left] + dv0[right] + dv0[up] + dv0[down]) * 0.25f;

      float frac = (Ix[pos] * sumU + Iy[pos] * sumV + Iz[pos]) /
                   (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

      du1[pos] = sumU - Ix[pos] * frac;
      dv1[pos] = sumV - Iy[pos] * frac;
    }
  }
}

void ComputeFlowGold(const float *I0, const float *I1, int width, int height,
                     int stride, float alpha, int nLevels, int nWarpIters,
                     int nSolverIters, float *u, float *v) {
  printf("Computing optical flow on CPU...\n");

  float *u0 = u;
  float *v0 = v;

  const float **pI0 = new const float *[nLevels];
  const float **pI1 = new const float *[nLevels];

  int *pW = new int[nLevels];
  int *pH = new int[nLevels];
  int *pS = new int[nLevels];

  const int pixelCountAligned = height * stride;

  float *tmp = new float[pixelCountAligned];
  float *du0 = new float[pixelCountAligned];
  float *dv0 = new float[pixelCountAligned];
  float *du1 = new float[pixelCountAligned];
  float *dv1 = new float[pixelCountAligned];
  float *Ix = new float[pixelCountAligned];
  float *Iy = new float[pixelCountAligned];
  float *Iz = new float[pixelCountAligned];
  float *nu = new float[pixelCountAligned];
  float *nv = new float[pixelCountAligned];

  int currentLevel = nLevels - 1;
  pI0[currentLevel] = I0;
  pI1[currentLevel] = I1;

  pW[currentLevel] = width;
  pH[currentLevel] = height;
  pS[currentLevel] = stride;

  for (; currentLevel > 0; --currentLevel) {
    int nw = pW[currentLevel] / 2;
    int nh = pH[currentLevel] / 2;
    int ns = iAlignUp(nw);
    pI0[currentLevel - 1] = new float[ns * nh];
    pI1[currentLevel - 1] = new float[ns * nh];

    Downscale(pI0[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI0[currentLevel - 1]);
    Downscale(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI1[currentLevel - 1]);

    pW[currentLevel - 1] = nw;
    pH[currentLevel - 1] = nh;
    pS[currentLevel - 1] = ns;
  }

  memset(u, 0, stride * height * sizeof(float));
  memset(v, 0, stride * height * sizeof(float));

  for (; currentLevel < nLevels; ++currentLevel) {
    for (int warpIter = 0; warpIter < nWarpIters; ++warpIter) {
      memset(du0, 0, pixelCountAligned * sizeof(float));
      memset(dv0, 0, pixelCountAligned * sizeof(float));
      memset(du1, 0, pixelCountAligned * sizeof(float));
      memset(dv1, 0, pixelCountAligned * sizeof(float));

      WarpImage(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
                pS[currentLevel], u, v, tmp);

      ComputeDerivatives(pI0[currentLevel], tmp, pW[currentLevel],
                         pH[currentLevel], pS[currentLevel], Ix, Iy, Iz);

      for (int iter = 0; iter < nSolverIters; ++iter) {
        SolveForUpdate(du0, dv0, Ix, Iy, Iz, pW[currentLevel], pH[currentLevel],
                       pS[currentLevel], alpha, du1, dv1);
        Swap(du0, du1);
        Swap(dv0, dv1);
      }

      for (int i = 0; i < pH[currentLevel] * pS[currentLevel]; ++i) {
        u[i] += du0[i];
        v[i] += dv0[i];
      }
    }

    if (currentLevel != nLevels - 1) {
      float scaleX = (float)pW[currentLevel + 1] / (float)pW[currentLevel];
      Upscale(u, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleX, nu);

      float scaleY = (float)pH[currentLevel + 1] / (float)pH[currentLevel];
      Upscale(v, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleY, nv);

      Swap(u, nu);
      Swap(v, nv);
    }
  }

  if (u != u0) {
    memcpy(u0, u, pixelCountAligned * sizeof(float));
    memcpy(v0, v, pixelCountAligned * sizeof(float));
    Swap(u, nu);
    Swap(v, nv);
  }

  for (int i = 0; i < nLevels - 1; ++i) {
    delete[] pI0[i];
    delete[] pI1[i];
  }

  delete[] pI0;
  delete[] pI1;
  delete[] pW;
  delete[] pH;
  delete[] pS;
  delete[] tmp;
  delete[] du0;
  delete[] dv0;
  delete[] du1;
  delete[] dv1;
  delete[] Ix;
  delete[] Iy;
  delete[] Iz;
  delete[] nu;
  delete[] nv;
}
