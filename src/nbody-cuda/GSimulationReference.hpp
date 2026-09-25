#include <cmath>

void accelerate_particles_ref( Particle* p, const int n, const float kSofteningSquared, const float kG )
{
  for (int i = 0; i < n; i++) {
    auto pi = p[i];
    RealType acc0 = pi.acc[0];
    RealType acc1 = pi.acc[1];
    RealType acc2 = pi.acc[2];
    for (int j = 0; j < n; j++) {
      RealType dx, dy, dz;
      RealType distance_sqr = 0.0f;
      RealType distance_inv = 0.0f;

      auto pj = p[j];
      dx = pj.pos[0] - pi.pos[0];  // 1flop
      dy = pj.pos[1] - pi.pos[1];  // 1flop
      dz = pj.pos[2] - pi.pos[2];  // 1flop

      distance_sqr =
        dx * dx + dy * dy + dz * dz + kSofteningSquared;  // 6flops
      distance_inv = 1.0f / std::sqrt(distance_sqr);       // 1div+1sqrt

      acc0 += dx * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
      acc1 += dy * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
      acc2 += dz * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
    }
    pi.acc[0] = acc0;
    pi.acc[1] = acc1;
    pi.acc[2] = acc2;
    p[i] = pi;
  }
}

void
update_particles_ref(Particle *__restrict__ p, RealType *__restrict__ e, const int n, RealType dt)
{
  for (int i = 0; i < n; i++) {
    auto pi = p[i];

    pi.vel[0] += pi.acc[0] * dt;  // 2flops
    pi.vel[1] += pi.acc[1] * dt;  // 2flops
    pi.vel[2] += pi.acc[2] * dt;  // 2flops

    pi.pos[0] += pi.vel[0] * dt;  // 2flops
    pi.pos[1] += pi.vel[1] * dt;  // 2flops
    pi.pos[2] += pi.vel[2] * dt;  // 2flops

    pi.acc[0] = 0.f;
    pi.acc[1] = 0.f;
    pi.acc[2] = 0.f;

    e[i] = pi.mass *
      (pi.vel[0] * pi.vel[0] + pi.vel[1] * pi.vel[1] +
       pi.vel[2] * pi.vel[2]);  // 7flops

    p[i] = pi;
  }
}

void
accumulate_energy_ref(RealType *e, const int n)
{
  for (int i = 1; i < n; i++) e[0] += e[i];
}

bool GSimulation::Verify() {
  RealType dt = get_tstep();
  int n = get_npart();
  std::vector<RealType> energy(n, 0.f);
  // allocate particles
  particles_.resize(n);

  InitPos();
  InitVel();
  InitAcc();
  InitMass();

#ifdef DEBUG
  PrintHeader();
#endif

  constexpr float kSofteningSquared = 1e-3f;
  // prevents explosion in the case the particles are really close to each other
  constexpr float kG = 6.67259e-11f;
  Particle *p = particles_.data();;

  RealType *e = energy.data();

  int nsteps = get_nsteps();
  // Looping across integration steps
  for (int s = 1; s <= nsteps; ++s) {

    accelerate_particles_ref(p, n, kSofteningSquared, kG);
    update_particles_ref(p, e, n, dt);
    accumulate_energy_ref(e, n);

    ref_kenergy_ = 0.5 * energy[0];
  }  // end of the time step loop
  std::cout << "\n";
  bool ok = fabsf(kenergy_ - ref_kenergy_) < 1e-3f;
  if (rank_ >= 0) {
    printf("[rank %d] %s\n", rank_, ok ? "PASS" : "FAIL");
  } else {
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  return ok;
}
