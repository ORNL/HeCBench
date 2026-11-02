//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
#include "GSimulation.hpp"

/* Default Constructor for the GSimulation class which sets up the default
 * values for number of particles, number of integration steps, time steo and
 * sample frequency */
GSimulation::GSimulation() {
  std::cout << "==============================="
            << "\n";
  std::cout << " Initialize Gravity Simulation"
            << "\n";
  set_npart(16000);
  set_nsteps(10);
  set_tstep(0.1);
  set_sfreq(1);
}

/* Set the number of particles */
void GSimulation::SetNumberOfParticles(int N) { set_npart(N); }

/* Set the number of integration steps */
void GSimulation::SetNumberOfSteps(int N) { set_nsteps(N); }

/* Initialize the position of all the particles using random number generator
 * between 0 and 1.0 */
void GSimulation::InitPos() {
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].pos[0] = unif_d(gen);
    particles_[i].pos[1] = unif_d(gen);
    particles_[i].pos[2] = unif_d(gen);
  }
}

/* Initialize the velocity of all the particles using random number generator
 * between -1.0 and 1.0 */
void GSimulation::InitVel() {
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(-1.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].vel[0] = unif_d(gen) * 1.0e-3f;
    particles_[i].vel[1] = unif_d(gen) * 1.0e-3f;
    particles_[i].vel[2] = unif_d(gen) * 1.0e-3f;
  }
}

/* Initialize the acceleration of all the particles to 0 */
void GSimulation::InitAcc() {
  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].acc[0] = 0.f;
    particles_[i].acc[1] = 0.f;
    particles_[i].acc[2] = 0.f;
  }
}

/* Initialize the mass of all the particles using a random number generator
 * between 0 and 1 */
void GSimulation::InitMass() {
  RealType n = static_cast<RealType>(get_npart());
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(0.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].mass = n * unif_d(gen);
  }
}

/* This function does the simulation logic for Nbody */
void GSimulation::Start() {
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

  total_time_ = 0.;

  constexpr float kSofteningSquared = 1e-3f;
  // prevents explosion in the case the particles are really close to each other
  constexpr float kG = 6.67259e-11f;
  double gflops = 1e-9 * ((11. + 10.) * n * n + n * 19.);
  int nf = 0;
  double av = 0.0, dev = 0.0;

  // Create a queue to the selected device and enabled asynchronous exception
  // handling for that queue
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  sycl::range<1> gws ((n+255)/256 * 256);
  sycl::range<1> lws (256);

  Particle *p = sycl::malloc_device<Particle>(n, q);
  q.memcpy(p, particles_.data(), n * sizeof(Particle)).wait();

  RealType *e = sycl::malloc_device<RealType>(n, q);

  int nsteps = get_nsteps();

  TimeInterval t0;
  // Looping across integration steps
  for (int s = 1; s <= nsteps; ++s) {
    TimeInterval ts0;

    // Submitting first kernel to device which computes acceleration of all
    // particles
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class compute_acceleration>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i >= n) return;

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
          distance_inv = sycl::rsqrt(distance_sqr);       // 1div+1sqrt

          auto strength = kG * pj.mass * distance_inv * distance_inv * distance_inv; // 4flops
          acc0 += dx * strength;  // 2flops
          acc1 += dy * strength;  // 2flops
          acc2 += dz * strength;  // 2flops
        }
        pi.acc[0] = acc0;
        pi.acc[1] = acc1;
        pi.acc[2] = acc2;
        p[i] = pi;
      });
    });
    // Second kernel updates the velocity and position for all particles
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class update_velocity_position>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        auto i = item.get_global_id(0);
        if (i >= n) return;

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
      });
    });
    /* Third kernel accumulates the energy of this Nbody system
     * Reduction operation can be done using reducer interface in SYCL 2020
     */
    q.submit([&](sycl::handler& h) {
      h.single_task<class accumulate_energy>([=]() {
        for (int i = 1; i < n; i++) e[0] += e[i];
      });
    });

    q.wait();
    double elapsed_seconds = ts0.Elapsed();

    q.memcpy(energy.data(), e, sizeof(RealType)).wait();

    kenergy_ = 0.5 * energy[0];
    if ((s % get_sfreq()) == 0) {
      nf += 1;
#ifdef DEBUG
      std::cout << " " << std::left << std::setw(8) << s << std::left
                << std::setprecision(5) << std::setw(8) << s * get_tstep()
                << std::left << std::setprecision(5) << std::setw(12)
                << kenergy_ << std::left << std::setprecision(5)
                << std::setw(12) << elapsed_seconds << std::left
                << std::setprecision(5) << std::setw(12)
                << gflops * get_sfreq() / elapsed_seconds << "\n";
#endif
      if (nf > 2) {
        av += gflops * get_sfreq() / elapsed_seconds;
        dev += gflops * get_sfreq() * gflops * get_sfreq() /
               (elapsed_seconds * elapsed_seconds);
      }
    }
  }  // end of the time step loop

  total_time_ = t0.Elapsed();
  total_flops_ = gflops * get_nsteps();
  av /= (double)(nf - 2);
  // when nf = 3, deviation is 0 (N/A)
  dev = (nf == 3) ? 0.0 : sqrt(dev / (double)(nf - 2) - av * av);

  std::cout << "\n";
  std::cout << "# Total Energy        : " << kenergy_ << "\n";
  std::cout << "# Total Time (s)      : " << total_time_ << "\n";
  std::cout << "# Average Performance : " << av << " +- " << dev << "\n";
  std::cout << "===============================\n";

  sycl::free(p, q);
  sycl::free(e, q);
}

#ifdef DEBUG
/* Print the headers for the output */
void GSimulation::PrintHeader() {
  std::cout << " nPart = " << get_npart() << "; "
            << "nSteps = " << get_nsteps() << "; "
            << "dt = " << get_tstep() << "\n";

  std::cout << "------------------------------------------------"
            << "\n";
  std::cout << " " << std::left << std::setw(8) << "s" << std::left
            << std::setw(8) << "dt" << std::left << std::setw(12) << "kenergy"
            << std::left << std::setw(12) << "time (s)" << std::left
            << std::setw(12) << "GFLOPS"
            << "\n";
  std::cout << "------------------------------------------------"
            << "\n";
}
#endif
