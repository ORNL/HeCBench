#include <cmath>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include "bude.h"

typedef std::chrono::high_resolution_clock::time_point TimePoint;

struct Params {

  size_t natlig;
  size_t natpro;
  size_t ntypes;
  size_t nposes;

  std::vector<Atom> protein;
  std::vector<Atom> ligand;
  std::vector<FFParams> forcefield;
  std::array<std::vector<float>, 6> poses;

  size_t iterations;

  //  size_t posesPerWI;
  size_t wgSize;
  std::string deckDir;

  friend std::ostream &operator<<(std::ostream &os, const Params &params) {
    os <<
      "natlig:      " << params.natlig << "\n" <<
      "natpro:      " << params.natpro << "\n" <<
      "ntypes:      " << params.ntypes << "\n" <<
      "nposes:      " << params.nposes << "\n" <<
      "iterations:  " << params.iterations << "\n" <<
      "posesPerWI:  " << NUM_TD_PER_THREAD << "\n" <<
      "wgSize:      " << params.wgSize << "\n";
    return os;
  }
};

SYCL_EXTERNAL
void fasten_main(
    sycl::nd_item<1> &item,
    FFParams *local_forcefield,
    size_t wgSize,
    size_t ntypes, size_t nposes,
    size_t natlig, size_t natpro,
    Atom *protein_molecule,
    Atom *ligand_molecule,
    float *transforms_0,
    float *transforms_1,
    float *transforms_2,
    float *transforms_3,
    float *transforms_4,
    float *transforms_5,
    FFParams *forcefield,
    float *etotals);

double elapsedMillis( const TimePoint &start, const TimePoint &end){
  auto elapsedNs = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
  return elapsedNs * 1e-6;
}

void printTimings(const Params &params, double millis) {

  // Average time per iteration
  double ms = (millis / params.iterations);
  double runtime = ms * 1e-3;

  // Compute FLOP/s
  double ops_per_wg = NUM_TD_PER_THREAD * 27 +
    params.natlig * (3 +
        NUM_TD_PER_THREAD * 18 +
        params.natpro * (11 + NUM_TD_PER_THREAD * 30)
        ) + NUM_TD_PER_THREAD;
  double total_ops = ops_per_wg * ((double) params.nposes / NUM_TD_PER_THREAD);
  double flops = total_ops / runtime;
  double gflops = flops / 1e9;

  double interactions = (double) params.nposes * (double) params.natlig * (double) params.natpro;
  double interactions_per_sec = interactions / runtime;

  // Print stats
  std::cout.precision(3);
  std::cout << std::fixed;
  std::cout << "- Total kernel time:    " << (millis) << " ms\n";
  std::cout << "- Average kernel time:   " << ms << " ms\n";
  std::cout << "- Interactions/s: " << (interactions_per_sec / 1e9) << " billion\n";
  std::cout << "- GFLOP/s:        " << gflops << "\n";
}

template<typename T>
std::vector<T> readNStruct(const std::string &path) {
  std::fstream s(path, std::ios::binary | std::ios::in);
  if (!s.good()) {
    throw std::invalid_argument("Bad file: " + path);
  }
  s.ignore(std::numeric_limits<std::streamsize>::max());
  auto len = s.gcount();
  s.clear();
  s.seekg(0, std::ios::beg);
  std::vector<T> xs(len / sizeof(T));
  s.read(reinterpret_cast<char *>(xs.data()), len);
  s.close();
  return xs;
}

Params loadParameters(const std::vector<std::string> &args) {

  Params params = {};

  // Defaults
  params.iterations = DEFAULT_ITERS;
  params.nposes = DEFAULT_NPOSES;
  params.wgSize = DEFAULT_WGSIZE;
  params.deckDir = DATA_DIR;
  //  params.posesPerWI = DEFAULT_PPWI;

  const auto readParam = [&args](size_t &current,
      const std::string &arg,
      const std::initializer_list<std::string> &matches,
      const std::function<void(std::string)> &handle) {
    if (matches.size() == 0) return false;
    if (std::find(matches.begin(), matches.end(), arg) != matches.end()) {
      if (current + 1 < args.size()) {
        current++;
        handle(args[current]);
      } else {
        std::cerr << "[";
        for (const auto &m : matches) std::cerr << m;
        std::cerr << "] specified but no value was given" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      return true;
    }
    return false;
  };

  const auto bindInt = [](const std::string &param, size_t &dest, const std::string &name) {
    try {
      auto parsed = std::stol(param);
      if (parsed < 0) {
        std::cerr << "positive integer required for <" << name << ">: `" << parsed << "`" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      dest = parsed;
    } catch (...) {
      std::cerr << "malformed value, integer required for <" << name << ">: `" << param << "`" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };

  for (size_t i = 0; i < args.size(); ++i) {
    using namespace std::placeholders;
    const auto arg = args[i];
    if (readParam(i, arg, {"--iterations", "-i"}, std::bind(bindInt, _1, std::ref(params.iterations), "iterations"))) continue;
    if (readParam(i, arg, {"--numposes", "-n"}, std::bind(bindInt, _1, std::ref(params.nposes), "numposes"))) continue;
    //    if (readParam(i, arg, {"--posesperwi", "-p"}, std::bind(bindInt, _1, std::ref(params.posesPerWI), "posesperwi"))) continue;
    if (readParam(i, arg, {"--wgsize", "-w"}, std::bind(bindInt, _1, std::ref(params.wgSize), "wgsize"))) continue;
    if (readParam(i, arg, {"--deck"}, [&](const std::string &param) { params.deckDir = param; })) continue;

    if (arg == "--help" || arg == "-h") {
      std::cout << "\n";
      std::cout << "Usage: ./main [OPTIONS]\n\n"
        << "Options:\n"
        << "  -h  --help               Print this message\n"
        << "  -i  --iterations I       Repeat kernel I times (default: " << DEFAULT_ITERS << ")\n"
        << "  -n  --numposes   N       Compute energies for N poses (default: " << DEFAULT_NPOSES << ")\n"
        //                << "  -p  --poserperwi PPWI    Compute PPWI poses per work-item (default: " << DEFAULT_PPWI << ")\n"
        << "  -w  --wgsize     WGSIZE  Run with work-group size WGSIZE using nd_range, set to 0 for plain range (default: " << DEFAULT_WGSIZE << ")\n"
        << "      --deck       DECK    Use the DECK directory as input deck (default: " << DATA_DIR << ")"
        << std::endl;
      std::exit(EXIT_SUCCESS);
    }

    std::cout << "Unrecognized argument '" << arg << "' (try '--help')" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  params.ligand = readNStruct<Atom>(params.deckDir + FILE_LIGAND);
  params.natlig = params.ligand.size();

  params.protein = readNStruct<Atom>(params.deckDir + FILE_PROTEIN);
  params.natpro = params.protein.size();

  params.forcefield = readNStruct<FFParams>(params.deckDir + FILE_FORCEFIELD);
  params.ntypes = params.forcefield.size();

  auto poses = readNStruct<float>(params.deckDir + FILE_POSES);
  if (poses.size() / 6 != params.nposes) {
    throw std::invalid_argument("Bad poses: " + std::to_string(poses.size()));
  }

  for (size_t i = 0; i < 6; ++i) {
    params.poses[i].resize(params.nposes);
    std::copy(
        std::next(poses.cbegin(), i * params.nposes),
        std::next(poses.cbegin(), i * params.nposes + params.nposes),
        params.poses[i].begin());

  }

  return params;
}

std::vector<float> runKernel(Params params) {

  const size_t wgSize = params.wgSize;
  const size_t ntypes = params.ntypes;
  const size_t nposes = params.nposes;
  const size_t natlig = params.natlig;
  const size_t natpro = params.natpro;

  std::vector<float> energies(nposes);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  Atom *protein = sycl::malloc_device<Atom>(natpro, q);
  q.memcpy(protein, params.protein.data(), sizeof(Atom) * natpro);

  Atom *ligand = sycl::malloc_device<Atom>(natlig, q);
  q.memcpy(ligand, params.ligand.data(), sizeof(Atom) * natlig);

  float *transforms_0 = sycl::malloc_device<float>(nposes, q);
  q.memcpy(transforms_0, params.poses[0].data(), sizeof(float) * nposes);

  float *transforms_1 = sycl::malloc_device<float>(nposes, q);
  q.memcpy(transforms_1, params.poses[1].data(), sizeof(float) * nposes);

  float *transforms_2 = sycl::malloc_device<float>(nposes, q);
  q.memcpy(transforms_2, params.poses[2].data(), sizeof(float) * nposes);

  float *transforms_3 = sycl::malloc_device<float>(nposes, q);
  q.memcpy(transforms_3, params.poses[3].data(), sizeof(float) * nposes);

  float *transforms_4 = sycl::malloc_device<float>(nposes, q);
  q.memcpy(transforms_4, params.poses[4].data(), sizeof(float) * nposes);

  float *transforms_5 = sycl::malloc_device<float>(nposes, q);
  q.memcpy(transforms_5, params.poses[5].data(), sizeof(float) * nposes);

  FFParams *forcefield = sycl::malloc_device<FFParams>(ntypes, q);
  q.memcpy(forcefield, params.forcefield.data(), sizeof(FFParams) * ntypes);

  float *results = sycl::malloc_device<float>(nposes, q);

  size_t global = ceil((nposes) / static_cast<double> (NUM_TD_PER_THREAD));
  global = wgSize * ceil(static_cast<double> (global) / wgSize);

  sycl::range<1> gws (global);
  sycl::range<1> lws (wgSize);

  auto kFn = [&](sycl::handler &h) {
    sycl::local_accessor<FFParams, 1> s_forcefield(sycl::range<1>(ntypes), h);
    h.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      fasten_main(item,
                  s_forcefield.get_multi_ptr<sycl::access::decorated::no>().get(),
                  wgSize,
                  ntypes,
                  nposes,
                  natlig,
                  natpro,
                  protein,
                  ligand,
                  transforms_0,
                  transforms_1,
                  transforms_2,
                  transforms_3,
                  transforms_4,
                  transforms_5,
                  forcefield,
                  results);
    });
  };

  // warmup
  q.submit(kFn).wait();

  auto kernelStart = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < params.iterations; ++i) {
    q.submit(kFn);
  }
  q.wait();
  auto kernelEnd = std::chrono::high_resolution_clock::now();

  q.memcpy(energies.data(), results, nposes*sizeof(float)).wait();

  printTimings(params, elapsedMillis(kernelStart, kernelEnd));

  sycl::free(protein, q);
  sycl::free(ligand, q);
  sycl::free(transforms_0, q);
  sycl::free(transforms_1, q);
  sycl::free(transforms_2, q);
  sycl::free(transforms_3, q);
  sycl::free(transforms_4, q);
  sycl::free(transforms_5, q);
  sycl::free(forcefield, q);
  sycl::free(results, q);

  return energies;
}

int main(int argc, char *argv[]) {

  auto args = std::vector<std::string>(argv + 1, argv + argc);
  auto params = loadParameters(args);

  std::cout << "Poses     : " << params.nposes << std::endl;
  std::cout << "Iterations: " << params.iterations << std::endl;
  std::cout << "Ligands   : " << params.natlig << std::endl;
  std::cout << "Proteins  : " << params.natpro << std::endl;
  std::cout << "Deck      : " << params.deckDir << std::endl;
  std::cout << "WG        : " << params.wgSize << std::endl;
  auto energies = runKernel(params);

#ifdef DUMP
  //XXX Keep the output format consistent with the C impl. so no fancy streams here
  FILE *output = fopen("result.out", "w+");

  printf("\nEnergies\n");
  for (size_t i = 0; i < params.nposes; i++) {
    fprintf(output, "%7.2f\n", energies[i]);
    if (i < 16)
      printf("%7.2f\n", energies[i]);
  }
  fclose(output);
#endif

  // Validate energies
  std::ifstream refEnergies(params.deckDir + FILE_REF_ENERGIES);
  size_t nRefPoses = params.nposes;
  if (params.nposes > REF_NPOSES) {
    std::cout << "Only validating the first " << REF_NPOSES << " poses.\n";
    nRefPoses = REF_NPOSES;
  }

  std::string line;
  float maxdiff = 0.0f;
  for (size_t i = 0; i < nRefPoses; i++) {
    if (!std::getline(refEnergies, line)) {
      throw std::logic_error("ran out of ref energies lines to verify");
    }
    float e = std::stof(line);
    if (std::fabs(e) < 1.f && std::fabs(energies[i]) < 1.f) continue;

    float diff = std::fabs(e - energies[i]) / e;
    if (diff > maxdiff) maxdiff = diff;
  }
  std::cout << "Largest difference was " <<
    std::setprecision(3) << (100 * maxdiff)
    << "%.\n\n"; // Expect numbers to be accurate to 2 decimal places
  refEnergies.close();

  return 0;
}
