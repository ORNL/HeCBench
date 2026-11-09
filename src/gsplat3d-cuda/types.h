#pragma once

// glm::vec2 / glm::vec3 layouts used by gsplat (no aligned-gentype padding).
struct vec2 {
    float x, y;
};

struct vec3 {
    float x, y, z;
};

static_assert(sizeof(vec2) == 8, "vec2 must match glm::vec2");
static_assert(sizeof(vec3) == 12, "vec3 must match glm::vec3");

static constexpr int kTileSize = 16;
static constexpr int kCdim = 3;
// Lanes that reduce together. This is not necessarily the hardware wave or
// sub-group size. NVIDIA warps and RDNA waves are 32 lanes, so one wave is
// one group. CDNA waves, including gfx942, are 64 lanes, so one wave holds
// two of these groups and the two groups run independently.
static constexpr int kWaveSize = 32;
static constexpr float kAlphaThreshold = 1.f / 255.f;
