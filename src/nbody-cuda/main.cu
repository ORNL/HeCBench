//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include "GSimulation.hpp"
#include "GSimulationReference.hpp"

int main(int argc, char** argv) {
  int n;      // number of particles
  int nstep;  // number ot integration steps

  GSimulation sim;

  if (argc > 1) {
    n = std::atoi(argv[1]);
    sim.SetNumberOfParticles(n);
    if (argc == 3) {
      nstep = std::atoi(argv[2]);
      if (nstep < 3) {
        std::cerr << "The number of integration steps should be at least 3.\n";
        return 1;
      }
      sim.SetNumberOfSteps(nstep);
    }
  }

  if (!sim.Verify())
    return 1;
  sim.Start();
  return 0;
}
