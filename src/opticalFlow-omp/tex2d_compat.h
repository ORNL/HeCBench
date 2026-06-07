// tex2d_compat.h — Portable tex2D for OpenMP target

#pragma once

struct Tex2DDesc {
  int width;
  int height;
  int pitch;
};

#pragma omp declare target
inline int tex2d_mirror_clamp(int i, int size) {
  int mi = ((i % (2 * size)) + 2 * size) % (2 * size);
  return (mi >= size) ? (2 * size - 1 - mi) : mi;
}

inline float tex2D_sw(const Tex2DDesc &tex, const float *ptr, float u, float v) {
  float fx = u * (float)tex.width - 0.5f;
  float fy = v * (float)tex.height - 0.5f;

  int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
  float tx = fx - (float)x0, ty = fy - (float)y0;

  int cx0 = tex2d_mirror_clamp(x0, tex.width);
  int cx1 = tex2d_mirror_clamp(x0 + 1, tex.width);
  int cy0 = tex2d_mirror_clamp(y0, tex.height);
  int cy1 = tex2d_mirror_clamp(y0 + 1, tex.height);

  float v00 = ptr[cy0 * tex.pitch + cx0];
  float v10 = ptr[cy0 * tex.pitch + cx1];
  float v01 = ptr[cy1 * tex.pitch + cx0];
  float v11 = ptr[cy1 * tex.pitch + cx1];

  return (1.f - tx) * (1.f - ty) * v00 + tx * (1.f - ty) * v10 +
         (1.f - tx) * ty * v01 + tx * ty * v11;
}

#pragma omp end declare target

#define TEX_TYPE Tex2DDesc
#define TEX_FETCH(t,p,x,y) tex2D_sw(t,p,x,y)
#define TEX_CREATE(v,w,h,s) Tex2DDesc v{w,h,s}
#define TEX_DESTROY(v)
