//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef _GSIMULATION_HPP
#define _GSIMULATION_HPP

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include "Particle.hpp"

class GSimulation {
 public:
  GSimulation();

  void Init();
  void SetNumberOfParticles(int N);
  void SetNumberOfSteps(int N);
  void SetRank(int rank) { rank_ = rank; }
  void Start();
  bool Verify();

 private:
  //  Particle *particles_;
  std::vector<Particle> particles_;
  int npart_;       // number of particles
  int nsteps_;      // number of integration steps
  RealType tstep_;  // time step of the simulation

  int sfreq_;  // sample frequency

  RealType kenergy_;  // kinetic energy
  RealType ref_kenergy_;  // kinetic energy

  double total_time_;   // total time of the simulation
  double total_flops_;  // total number of FLOPS
  int rank_ = -1;       // MPI replica rank, or -1 for a non-MPI build

  void InitPos();
  void InitVel();
  void InitAcc();
  void InitMass();

  void set_npart(const int &N) { npart_ = N; }
  int get_npart() const { return npart_; }

  void set_tstep(const RealType &dt) { tstep_ = dt; }
  RealType get_tstep() const { return tstep_; }

  void set_nsteps(const int &n) { nsteps_ = n; }
  int get_nsteps() const { return nsteps_; }

  void set_sfreq(const int &sf) { sfreq_ = sf; }
  int get_sfreq() const { return sfreq_; }

  void PrintHeader();
};

// The TimeInterval is a simple RAII class.
// Construct the timer at the point you want to start timing.
// Use the Elapsed() method to return time since construction.

class TimeInterval {
 public:
  TimeInterval() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

 private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

#endif
