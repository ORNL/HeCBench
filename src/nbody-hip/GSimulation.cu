//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <hip/hip_runtime.h>
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
#include <mpi.h>
#endif
#include "GSimulation.hpp"
#include "GSimulationKernels.hpp"

#define CHECK_HIP_ERROR(status)                                               \
  do {                                                                        \
    hipError_t error = (status);                                              \
    if (error != hipSuccess) {                                                \
      std::cerr << "HIP error: '" << hipGetErrorString(error) << "'("       \
                << error << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
      /* A failed rank must not leave peers in the final reduction. */         \
      fatal_exit();                                                           \
    }                                                                         \
  } while (0)

static void fatal_exit() {
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  exit(EXIT_FAILURE);
}

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

  Particle *p;
  CHECK_HIP_ERROR(hipMalloc((void**)&p, sizeof(Particle) * n));
  CHECK_HIP_ERROR(hipMemcpyAsync(p, particles_.data(), sizeof(Particle) * n,
                                 hipMemcpyHostToDevice, 0));

  RealType *e;
  CHECK_HIP_ERROR(hipMalloc((void**)&e, sizeof(RealType) * n));
  CHECK_HIP_ERROR(hipMemcpyAsync(e, energy.data(), sizeof(RealType) * n,
                                 hipMemcpyHostToDevice, 0));

  dim3 grids((n+255)/256);
  dim3 threads(256);

  TimeInterval t0;
  int nsteps = get_nsteps();
  // Looping across integration steps
  for (int s = 1; s <= nsteps; ++s) {
    TimeInterval ts0;

    accelerate_particles<<<grids, threads>>>(p, n, kSofteningSquared, kG);
    update_particles<<<grids, threads>>>(p, e, n, dt);
    accumulate_energy<<<1,1>>>(e, n);

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    double elapsed_seconds = ts0.Elapsed();

    CHECK_HIP_ERROR(
        hipMemcpy(energy.data(), e, sizeof(RealType), hipMemcpyDeviceToHost));

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
  if (rank_ >= 0) std::cout << "[rank " << rank_ << "] ";
  std::cout << "# Total Energy        : " << kenergy_ << "\n";
  if (rank_ >= 0) std::cout << "[rank " << rank_ << "] ";
  std::cout << "# Total Time (s)      : " << total_time_ << "\n";
  if (rank_ >= 0) std::cout << "[rank " << rank_ << "] ";
  std::cout << "# Average Performance : " << av << " +- " << dev << "\n";
  std::cout << "===============================\n";

  CHECK_HIP_ERROR(hipFree(p));
  CHECK_HIP_ERROR(hipFree(e));
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
