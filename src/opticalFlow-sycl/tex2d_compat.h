// tex2d_compat.h — Portable tex2D

#pragma once

// Descriptor passed to kernels in place of hipTextureObject_t
struct Tex2DDesc {
    const float* ptr;   // pitched device pointer
    int   width;        // image width in elements
    int   height;       // image height in elements
    int   pitch;        // row stride in elements (pitchInBytes / sizeof(float))
};

// Mirror repeat address clamping
inline int tex2d_mirror_clamp(int i, int size) {
    int mi = ((i % (2 * size)) + 2 * size) % (2 * size);
    return (mi >= size) ? (2 * size - 1 - mi) : mi;
}

// Software bilinear tex2D — normalized coords [0,1), mirror addressing, linear filter
// Matches: normalizedCoords=true, filterMode=Linear, addressMode=Mirror
inline float tex2D_sw(const Tex2DDesc& tex, float u, float v) {
    // Convert normalized coords to pixel space
    float fx = u * (float)tex.width  - 0.5f;
    float fy = v * (float)tex.height - 0.5f;

    int x0 = (int)sycl::floor(fx), y0 = (int)sycl::floor(fy);
    float tx = fx - (float)x0, ty = fy - (float)y0;

    int cx0 = tex2d_mirror_clamp(x0,   tex.width);
    int cx1 = tex2d_mirror_clamp(x0+1, tex.width);
    int cy0 = tex2d_mirror_clamp(y0,   tex.height);
    int cy1 = tex2d_mirror_clamp(y0+1, tex.height);

    float v00 = tex.ptr[cy0 * tex.pitch + cx0];
    float v10 = tex.ptr[cy0 * tex.pitch + cx1];
    float v01 = tex.ptr[cy1 * tex.pitch + cx0];
    float v11 = tex.ptr[cy1 * tex.pitch + cx1];

    return (1.f-tx)*(1.f-ty)*v00 + tx*(1.f-ty)*v10
         + (1.f-tx)*ty      *v01 + tx*ty      *v11;
}
