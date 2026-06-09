//   Pass 1 (guidance):     weighted average color per output patch
//   Pass 2 (downsampling): L2-distance reweighted downsampling with lambda

#ifndef DPID_REF_TYPES_DEFINED
#define DPID_REF_TYPES_DEFINED

struct ref_uchar3 { unsigned char x, y, z; };
struct ref_float4 { float x, y, z, w; };

static inline ref_uchar3 make_ref_uchar3(unsigned char x, unsigned char y,
                                          unsigned char z) {
  ref_uchar3 v; v.x = x; v.y = y; v.z = z; return v;
}
static inline ref_float4 make_ref_float4(float x, float y, float z, float w) {
  ref_float4 v; v.x = x; v.y = y; v.z = z; v.w = w; return v;
}

#endif // DPID_REF_TYPES_DEFINED

static inline void ref_normalize(ref_float4& v) {
  v.x /= v.w; v.y /= v.w; v.z /= v.w; v.w = 1.f;
}

static inline void ref_add(ref_float4& out, unsigned char cx, unsigned char cy,
                            unsigned char cz, float f) {
  out.x += cx * f; out.y += cy * f; out.z += cz * f; out.w += f;
}

static inline float ref_contribution(float sx, float ex, float sy, float ey,
                                     float f, uint32_t x, uint32_t y) {
  if (x        <  sx) f *= 1.f - (sx - x);
  if ((x+1.f)  >  ex) f *= 1.f - ((x+1.f) - ex);
  if (y        <  sy) f *= 1.f - (sy - y);
  if ((y+1.f)  >  ey) f *= 1.f - ((y+1.f) - ey);
  return f;
}

static inline float ref_lambda(float lambda_val, float dist) {
  if (lambda_val == 0.f) return 1.f;
  if (lambda_val == 1.f) return dist;
  return powf(dist, lambda_val);
}

static inline float ref_l2dist(const ref_float4& avg,
                                unsigned char cx, unsigned char cy,
                                unsigned char cz) {
  float dx = avg.x - cx, dy = avg.y - cy, dz = avg.z - cz;
  return std::sqrt(dx*dx + dy*dy + dz*dz) / 441.6729559f;
}

static void ref_guidance(const unsigned char* input_bytes,  // uchar3 array
                          unsigned char* patches_bytes,      // uchar3 array
                          uint32_t oWidth, uint32_t oHeight,
                          uint32_t iWidth,
                          float pWidth, float pHeight) {
  for (uint32_t py = 0; py < oHeight; py++) {
    for (uint32_t px = 0; px < oWidth; px++) {
      float sx = std::fmax( px    * pWidth,  0.f);
      float ex = std::fmin((px+1) * pWidth,  (float)iWidth);
      float sy = std::fmax( py    * pHeight, 0.f);
      float ey = std::fmin((py+1) * pHeight, (float)(oHeight * pHeight));

      uint32_t sxr = (uint32_t)std::floor(sx), syr = (uint32_t)std::floor(sy);
      uint32_t exr = (uint32_t)std::ceil(ex),  eyr = (uint32_t)std::ceil(ey);
      uint32_t xCount = exr - sxr, yCount = eyr - syr;

      ref_float4 color = make_ref_float4(0.f, 0.f, 0.f, 0.f);

      for (uint32_t i = 0; i < xCount * yCount; i++) {
        uint32_t x = sxr + (i % xCount), y = syr + (i / xCount);
        const unsigned char* pix = input_bytes + (x + y * iWidth) * 3;
        float f = ref_contribution(sx, ex, sy, ey, 1.f, x, y);
        ref_add(color, pix[0], pix[1], pix[2], f);
      }

      ref_normalize(color);
      unsigned char* out = patches_bytes + (px + py * oWidth) * 3;
      out[0] = (unsigned char)color.x;
      out[1] = (unsigned char)color.y;
      out[2] = (unsigned char)color.z;
    }
  }
}

static ref_float4 ref_calc_average(const unsigned char* patches_bytes,
                                    uint32_t oWidth, uint32_t oHeight,
                                    uint32_t px, uint32_t py) {
  const float corner = 1.f, edge = 2.f, center = 4.f;
  ref_float4 avg = make_ref_float4(0.f, 0.f, 0.f, 0.f);

  auto add_patch = [&](uint32_t qx, uint32_t qy, float w) {
    const unsigned char* p = patches_bytes + (qx + qy * oWidth) * 3;
    ref_add(avg, p[0], p[1], p[2], w);
  };

  if (py > 0) {
    if (px > 0) add_patch(px-1, py-1, corner);
    add_patch(px, py-1, edge);
    if (px+1 < oWidth) add_patch(px+1, py-1, corner);
  }
  if (px > 0) add_patch(px-1, py, edge);
  add_patch(px, py, center);
  if (px+1 < oWidth)  add_patch(px+1, py, edge);
  if (py+1 < oHeight) {
    if (px > 0) add_patch(px-1, py+1, corner);
    add_patch(px, py+1, edge);
    if (px+1 < oWidth) add_patch(px+1, py+1, corner);
  }

  ref_normalize(avg);
  return avg;
}

static void ref_downsampling(const unsigned char* input_bytes,
                              const unsigned char* patches_bytes,
                              unsigned char* output_bytes,
                              uint32_t oWidth, uint32_t oHeight,
                              uint32_t iWidth,
                              float pWidth, float pHeight,
                              float lambda_val) {
  for (uint32_t py = 0; py < oHeight; py++) {
    for (uint32_t px = 0; px < oWidth; px++) {
      float sx = std::fmax( px    * pWidth,  0.f);
      float ex = std::fmin((px+1) * pWidth,  (float)iWidth);
      float sy = std::fmax( py    * pHeight, 0.f);
      float ey = std::fmin((py+1) * pHeight, (float)(oHeight * pHeight));

      uint32_t sxr = (uint32_t)std::floor(sx), syr = (uint32_t)std::floor(sy);
      uint32_t exr = (uint32_t)std::ceil(ex),  eyr = (uint32_t)std::ceil(ey);
      uint32_t xCount = exr - sxr, yCount = eyr - syr;

      ref_float4 avg = ref_calc_average(patches_bytes, oWidth, oHeight, px, py);
      ref_float4 color = make_ref_float4(0.f, 0.f, 0.f, 0.f);

      for (uint32_t i = 0; i < xCount * yCount; i++) {
        uint32_t x = sxr + (i % xCount), y = syr + (i / xCount);
        const unsigned char* pix = input_bytes + (x + y * iWidth) * 3;
        float f = ref_l2dist(avg, pix[0], pix[1], pix[2]);
        f = ref_lambda(lambda_val, f);
        f = ref_contribution(sx, ex, sy, ey, f, x, y);
        ref_add(color, pix[0], pix[1], pix[2], f);
      }

      unsigned char* out = output_bytes + (px + py * oWidth) * 3;
      if (color.w == 0.f) {
        out[0] = (unsigned char)avg.x;
        out[1] = (unsigned char)avg.y;
        out[2] = (unsigned char)avg.z;
      } else {
        ref_normalize(color);
        out[0] = (unsigned char)color.x;
        out[1] = (unsigned char)color.y;
        out[2] = (unsigned char)color.z;
      }
    }
  }
}

// hInput and hOutput are arrays of uchar3 (3 bytes per pixel, packed).
// Call as: reference(p, hInput, hOutput)
// where hInput/hOutput are cast to void* from uchar3* in the caller.

void reference(uint32_t oWidth, uint32_t oHeight,
               uint32_t iWidth, uint32_t iHeight,
               float pWidth, float pHeight,
               float lambda_val,
               const void* hInput, void* hOutput) {
  unsigned char* patches =
      (unsigned char*) malloc(oWidth * oHeight * 3);

  ref_guidance((const unsigned char*)hInput, patches,
               oWidth, oHeight, iWidth, pWidth, pHeight);

  ref_downsampling((const unsigned char*)hInput, patches,
                   (unsigned char*)hOutput,
                   oWidth, oHeight, iWidth, pWidth, pHeight, lambda_val);

  free(patches);
}
