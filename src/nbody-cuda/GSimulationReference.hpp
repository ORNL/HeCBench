#include <cmath>
#include <iomanip>
#include <sstream>

#include "nbody_constants.hpp"

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

// Short integrations stay in the rounding regime. A long run does not:
// the same two correct integrators drift apart as the step count grows.
constexpr int kFixedVerifyParticles = 2048;
constexpr int kFixedVerifySteps = 10;
constexpr float kRelativeTolerance = 4e-6f;

RealType GSimulation::ReferenceEnergy(int n, int steps) {
  set_npart(n);
  particles_.resize(n);
  InitPos();
  InitVel();
  InitAcc();
  InitMass();

  const RealType dt = get_tstep();
  std::vector<RealType> energy(n, 0.f);
  Particle *p = particles_.data();
  RealType *e = energy.data();

  for (int s = 0; s < steps; ++s) {
    accelerate_particles_ref(p, n, kSofteningSquared, kG);
    update_particles_ref(p, e, n, dt);
    accumulate_energy_ref(e, n);
  }
  return 0.5f * e[0];
}

// Device and host each start from the seeded initial conditions.
// Both results must be finite and agree to kRelativeTolerance.
bool GSimulation::Verify() {
  const int npart = get_npart();
  const int cases[][2] = {{npart, 1}, {kFixedVerifyParticles, kFixedVerifySteps}};
  bool ok = true;

  for (const auto &check : cases) {
    const int n = check[0];
    const int steps = check[1];
    const RealType device = Integrate(n, steps);
    const RealType reference = ReferenceEnergy(n, steps);
    const float difference = fabsf(device - reference);
    const bool finite = std::isfinite(device) && std::isfinite(reference);
    const bool close = difference <= kRelativeTolerance * fabsf(reference);
    if (!finite || !close) {
      ok = false;
      std::ostringstream message;
      message << std::setprecision(9) << "n=" << n << " steps=" << steps
              << " device=" << device << " reference=" << reference;
      if (!finite) {
        message << " non-finite";
      } else if (reference == 0.f) {
        message << " rel=inf";
      } else {
        const double relative =
            std::fabs(static_cast<double>(device) - reference) /
            std::fabs(static_cast<double>(reference));
        message << std::scientific << std::setprecision(3)
                << " rel=" << relative;
      }
      std::cout << message.str() << "\n";
    }
  }

  set_npart(npart);
  std::cout << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}
