//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
#include <hip/hip_runtime.h>
#include <mpi.h>
#endif
#include "GSimulation.hpp"
#include "GSimulationReference.hpp"

int main(int argc, char** argv) {
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
  MPI_Init(&argc, &argv);
  int rank, rank_count, local_rank;
  MPI_Comm local_comm;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                      MPI_INFO_NULL, &local_comm);
  MPI_Comm_rank(local_comm, &local_rank);

  int visible_devices = 0;
  if (hipGetDeviceCount(&visible_devices) != hipSuccess || visible_devices < 1) {
    std::cerr << "[rank " << rank << "] No visible HIP device\n";
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  int selected_device = visible_devices == 1 ? 0 : local_rank;
  if (selected_device >= visible_devices) {
    std::cerr << "[rank " << rank << "] local rank " << local_rank
              << " cannot be mapped to " << visible_devices
              << " visible HIP devices\n";
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (hipSetDevice(selected_device) != hipSuccess) {
    std::cerr << "[rank " << rank << "] Failed to select HIP device "
              << selected_device << "\n";
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  char processor[MPI_MAX_PROCESSOR_NAME];
  int processor_length;
  MPI_Get_processor_name(processor, &processor_length);
  std::cout << "[rank " << rank << "/" << rank_count << " local_rank "
            << local_rank << "] host=" << processor << " device="
            << selected_device << " visible_devices=" << visible_devices
            << "\n";
  MPI_Comm_free(&local_comm);
#endif

  int n;      // number of particles
  int nstep;  // number ot integration steps

  GSimulation sim;
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
  sim.SetRank(rank);
#endif

  if (argc > 1) {
    n = std::atoi(argv[1]);
    sim.SetNumberOfParticles(n);
    if (argc == 3) {
      nstep = std::atoi(argv[2]);
      if (nstep < 3) {
        std::cerr << "The number of integration steps should be at least 3\n";
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
        MPI_Finalize();
#endif
        return 1;
      }
      sim.SetNumberOfSteps(nstep);
    }
  }

  sim.Start();
  [[maybe_unused]] bool ok = sim.Verify();

#ifdef HECBENCH_ENABLE_MPI_REPLICAS
  int local_pass = ok ? 1 : 0;
  int passed_count = 0;
  MPI_Allreduce(&local_pass, &passed_count, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "OVERALL "
              << (passed_count == rank_count ? "PASS" : "FAIL") << " ("
              << passed_count << "/" << rank_count << " replica ranks)\n";
  }
  MPI_Finalize();
  return passed_count == rank_count ? 0 : 2;
#endif

  return 0;
}
