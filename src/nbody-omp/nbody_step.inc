// Included inside an OpenMP target data region that already maps p[0:n] and e[0:n].
// The enclosing scope provides Particle *p, RealType *e, int n, RealType dt,
// and kSofteningSquared and kG from nbody_constants.hpp.
#pragma omp target teams distribute parallel for thread_limit(256)
for (int i = 0; i < n; i++) {
  auto pi = p[i];
  RealType acc0 = pi.acc[0];
  RealType acc1 = pi.acc[1];
  RealType acc2 = pi.acc[2];

  for (int j = 0; j < n; j++) {
    auto pj = p[j];
    RealType dx = pj.pos[0] - pi.pos[0];
    RealType dy = pj.pos[1] - pi.pos[1];
    RealType dz = pj.pos[2] - pi.pos[2];
    RealType distance_sqr = dx * dx + dy * dy + dz * dz + kSofteningSquared;
    RealType distance_inv = 1.0f / sqrtf(distance_sqr);
    auto strength = kG * pj.mass * distance_inv * distance_inv * distance_inv;

    acc0 += dx * strength;
    acc1 += dy * strength;
    acc2 += dz * strength;
  }
  pi.acc[0] = acc0;
  pi.acc[1] = acc1;
  pi.acc[2] = acc2;
  p[i] = pi;
}

#pragma omp target teams distribute parallel for thread_limit(256)
for (int i = 0; i < n; i++) {
  auto pi = p[i];
  pi.vel[0] += pi.acc[0] * dt;
  pi.vel[1] += pi.acc[1] * dt;
  pi.vel[2] += pi.acc[2] * dt;

  pi.pos[0] += pi.vel[0] * dt;
  pi.pos[1] += pi.vel[1] * dt;
  pi.pos[2] += pi.vel[2] * dt;

  pi.acc[0] = 0.f;
  pi.acc[1] = 0.f;
  pi.acc[2] = 0.f;

  e[i] = pi.mass *
    (pi.vel[0] * pi.vel[0] + pi.vel[1] * pi.vel[1] + pi.vel[2] * pi.vel[2]);
  p[i] = pi;
}

#pragma omp target
for (int i = 1; i < n; i++) e[0] += e[i];
