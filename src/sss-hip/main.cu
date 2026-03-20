// Chiranjit Mukherjee  (chiranjit@soe.ucsc.edu)

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <chrono>

#define SSS     // Runs the Stochastic Shotgun Search
#define USE_GPU // If uncommented, runs kernels on GPU
#define MAXL 10 // Maximum number of mixture components that can be accommodated

#ifdef SSS
// SSS- runtime parameters
// #define maxLocalWastedIterations (n+p)  // In paper, C
#define maxLocalWastedIterations 50

#define climbDownStepSize 10               // In paper, D

// #define maxLocalJumpCount 10            // In paper, R
#define maxLocalJumpCount 3

#define MAXNGLOBALJUMP 2                   // In paper, S / (C * R)
// SSS- parameters for lists of models saved

#define sizeOfFeatureSelectionList 20      // In paper, M

#define sizeOfBestList 100 // Number of highest-score models to keep track of
// SSS-
#define LOCALMOVE_SFACTOR 0.001
#define GLOBALJUMP_SFACTOR 0.01
#define G_TO_XI int(L * p * (p - 1) / (2 * n)) // In paper, g
#define XI_TO_SM 10                            // In paper, h
#define LOOKFORWARD 5                          // In paper, f
#define RGMS_T 2                               // In paper, t
// number of chains parameters
#define N_INIT                                                                 \
  1 // Number of points of initial models provided by the user in folder DATA/

#define TRY_EACH_INIT                                                          \
  1 // Number of times to restart from each given initial point

#define N_RANDOM_RESTART                                                       \
  1 // Number of times to restart from random random initial points

#define N_MODES_LIST_RESTART 1 // Number of times to start from

// #define maxNmodes
// ((TRY_EACH_INIT*N_INIT+N_RANDOM_RESTART+N_MODES_LIST_RESTART)+1)
#define maxNmodes 2
#endif

#define PI 3.1415926
#define log_2 0.693147180559945
#define log_pi_over_4 0.286182471462350
#define log_2_pi 1.837877066409345
#define NEG_INF -999999.0
#define myBool bool
#define myInt short // Using short interger
#define myIntFactor 2
#define intFactor 4
// #define Real double
#define Real float // Using floating-point
#define ISFLOAT 10
using namespace std;

#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>
#define GSL_INTEGRATION_GRIDSIZE 1000
gsl_integration_workspace *w;
gsl_function F;

#include <gsl/gsl_randist.h>
#define RANDOMSEED 314159265

// Define hyperparameters for the prior distribution of (mu, K | G)
#define N0 0.01
#define DELTA0 3
#define JEFFREYS_PRIOR
gsl_rng *rnd;

#ifdef USE_GPU
#include <hip/hip_runtime.h>
#define BLOCK_SIZE 32
#define SYNC __syncthreads()
typedef struct {
  hipStream_t delete_stream;
  hipStream_t add_stream;
  myInt *d_in_delete;
  myInt *d_in_add;
  myInt *d_which_delete;
  myInt *d_which_add;
  myInt *h_in_delete;
  myInt *h_in_add;
  myInt *which_delete;
  myInt *which_add;
  int n_add, n_delete;
} MGPUstuff;
#else
typedef struct {
} MGPUstuff;
#endif
MGPUstuff device;

// Include source files
#include "utilities.cpp"
#ifndef GRAPH_CPP
#include "graph.cpp"
#endif
#ifndef GWISH_CPP
#include "gwish.cpp"
#endif
#ifndef DPMIXGGM_CPP
#include "DPmixGGM.cpp"
#endif
#ifndef LISTS_CPP
#include "DPmixGGM_Lists.cpp"
#endif
#ifndef SSSMOVES_CPP
#include "DPmixGGM_SSSmoves.cpp"
#endif

//////////////////////////////////////////////////////////////// START OF MAIN
//////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  // declarations and initialisations
  int i, j, l, q, r, t;
  int L = 2;
  long int k;
  Real score;
  char initID[] = {'1', '2', '3'};

  // Initializing gsl random variate generators and integration tools
  const gsl_rng_type *T;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  rnd = gsl_rng_alloc(T);
  gsl_rng_set(rnd, RANDOMSEED);
#ifdef SSS
  unsigned long int seedset[maxNmodes];
  for (i = 0; i < maxNmodes; i++) {
    seedset[i] = gsl_rng_get(rnd);
    printf("seed: %zu\n", seedset[i]);
  }
#endif

  w = gsl_integration_workspace_alloc(GSL_INTEGRATION_GRIDSIZE);

  // DATA INPUT
  char datafile[50] = "";
  strcpy(datafile, "DATA/");
  strcat(datafile, argv[1]);
  strcat(datafile, ".txt");
  ifstream data(datafile);
  int n, p;
  data >> n;
  data >> p;
  printf("n=%d p=%d\n", n, p);
  Real *X = new Real[n * p];
  for (i = 0; i < n; i++) {
    for (j = 0; j < p; j++) {
      data >> X[p * i + j];
    }
  }
  data.close();

  printf("Max number of modes = %d\n", maxNmodes);

  // more declarations and initialisations
  int ee = p * (p - 1) / 2;

////////////////////////////////////////////////////////////// START OF SSS
//////////////////////////////////////////////////////////////////
#ifdef SSS

  // OUTPUT FILES
  char outfile[100] = "";
  strcpy(outfile, "RES/");
#ifndef USE_GPU
  strcat(outfile, argv[1]);
  strcat(outfile, "_modes_CPU.txt");
  ofstream outmodes(outfile);
  outmodes << n << " " << p << endl;
#ifndef USE_GPU
  strcpy(outfile, "RES/");
  strcat(outfile, argv[1]);
  strcat(outfile, "_best_CPU.txt");
  ofstream outbest(outfile);
  outbest << n << " " << p << endl;
#endif
#else
  strcat(outfile, argv[1]);
  strcat(outfile, "_modes_GPU.txt");
  ofstream outmodes(outfile);
  outmodes << n << " " << p << endl;
#ifndef USE_GPU
  strcpy(outfile, "RES/");
  strcat(outfile, argv[1]);
  strcat(outfile, "_best_GPU.txt");
  ofstream outbest(outfile);
  outbest << n << " " << p << endl;
#endif
#endif

  // Initialisations
  State initstates[N_INIT + N_RANDOM_RESTART];
  int *initstateID = new int[maxNmodes];
  for (i = 0; i < N_INIT; i++) {
    strcpy(datafile, "DATA/");
    strcat(datafile, argv[1]);
    strcat(datafile, "_init");
    strncat(datafile, &initID[i], 1);
    strcat(datafile, ".txt");
    ifstream initfile(datafile);
    initstates[i] = new DPmixGGM(X, L, n, p, 0.1, initfile);
    initfile.close();
    initstateID[i] = i;
  }

  State state = new DPmixGGM(initstates[0]);
  State localBestState = new DPmixGGM(state);
  State globalBestState = new DPmixGGM(state);
  List featureList = new DPmixGGMlist(sizeOfFeatureSelectionList, n, p);
  List modesList = new DPmixGGMlist(maxNmodes, n, p);
#ifdef USE_GPU
  List bestList = (List)NULL;
#else
  List bestList = new DPmixGGMlist(sizeOfBestList, n, p);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef USE_GPU
  size_t size_temp;

  hipStreamCreate(&(device.delete_stream));
  hipStreamCreate(&(device.add_stream));

  size_temp = sizeof(myInt) * (3 + p + p * p + 2 * ee);
  hipMalloc((void **)&(device.d_in_delete), size_temp);

  size_temp = sizeof(myInt) * (3 + 4 * p + 2 * p * p + 2 * ee);
  hipMalloc((void **)&(device.d_in_add), size_temp);

  size_temp = sizeof(myInt) * ee;
  hipMalloc((void **)&(device.d_which_delete), size_temp);
  hipMalloc((void **)&(device.d_which_add), size_temp);

  device.h_in_delete = new myInt[3 + p + p * p + 2 * ee];
  device.h_in_add = new myInt[4 + 4 * p + 2 * p * p + 2 * ee];
  device.which_delete = new myInt[ee];
  device.which_add = new myInt[ee];
#endif

  // more declarations and initialisations
  bool globalMoveFlag = 0;
  myInt nmodes = 1;
  Real localBestScore = NEG_INF, globalBestScore = NEG_INF;
  gsl_rng_set(rnd, seedset[nmodes - 1]);
  int wastedIterations = 0;
  int localJumpCount = 0, globalJumpCount = 0;
  int num_cases = 0;
  long int num_allModels = 0;

  // initial xi scan
  num_cases += updateAllXis(1, state, bestList);
  L = state->L;
  score = state->plp;
  for (l = 0; l < L; l++) {
    score += state->pll[l];
  }
  k = 0;
  printf("initial: k=%ld L=%d score=%.4f localBestScore=%.4f globalBestScore=%.4f "
         "nmodes=%d num_cases=%d num_allModels=%ld\n",
         k, state->L, score, localBestScore, globalBestScore, nmodes,
         num_cases, num_allModels);

  // start the stopwatch
  auto start = std::chrono::steady_clock::now();
  while (nmodes <= maxNmodes) {
    k++;
    num_cases = 0;

    // LOCAL MOVES
    // ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if ((k % G_TO_XI)) {
      num_cases += updateOneEdgeInEveryG(state->L, NULL, 0, state->graphlist,
                                         state->pll, NULL, state, bestList);
    } else {
      j = k / G_TO_XI;
      if (j % XI_TO_SM) {
        if (state->L > 1) {
          num_cases += updateAllXis(1, state, bestList);
          num_cases += Merge(state, bestList, LOOKFORWARD, 0);
        }
      } else {
        num_cases += splitMerge(state, featureList, bestList, LOOKFORWARD,
                                LOCALMOVE_SFACTOR, 0, 1, RGMS_T);
      }
    }
    // LOCAL MOVES
    // ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // MODE BREAK MOVES
    // //////////////////////////////////////////////////////////////////////////////////////////////////////////
    if ((wastedIterations > maxLocalWastedIterations) &&
        (localJumpCount < maxLocalJumpCount)) {
      wastedIterations = 0;
      localJumpCount++;
      state->CopyState(localBestState);

      // local graph jump
      for (i = 0; i < localJumpCount; i++) {
        num_cases += updateOneEdgeInEveryG(
            state->L, NULL, (i + 1) * climbDownStepSize, state->graphlist,
            state->pll, NULL, state, bestList);
      }
    }
    // MODE BREAK MOVES
    // //////////////////////////////////////////////////////////////////////////////////////////////////////////

    // GLOBAL JUMP MOVES
    // /////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (wastedIterations > maxLocalJumpCount * maxLocalWastedIterations) {
      if (globalJumpCount == MAXNGLOBALJUMP) {
        globalMoveFlag = 1;
      } else {
        wastedIterations = 0;
        globalJumpCount++;
        state->CopyState(globalBestState);
        localBestScore = NEG_INF;
        num_cases +=
            globalJumpAllG(1, 1, LOOKFORWARD, GLOBALJUMP_SFACTOR, state,
                           featureList, bestList); // larger graph jump
        state->plp =
            state->partitionlogPrior(state->L, state->xi, state->alpha);
        for (l = 0; l < state->L; l++) {
          state->pll[l] =
              state->cluster_k_loglikelihood(l, state->xi, state->graphlist[l]);
        }
      }
    }
    // GLOBAL JUMP MOVES
    // /////////////////////////////////////////////////////////////////////////////////////////////////////////

    // SEARCH RESTART
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (globalMoveFlag) {
      globalMoveFlag = 0;
      modesList->UpdateList(globalBestState);
      nmodes++;
      gsl_rng_set(rnd, seedset[nmodes - 1]);
      k = 0;
      localBestScore = NEG_INF;
      globalBestScore = NEG_INF;
      featureList->FlushList(state);

#ifndef USE_GPU
      if (nmodes > maxNmodes) {
        break;
      }
#else
      if (nmodes > maxNmodes - 1) {
        break;
      }
#endif

      if (nmodes <=
          TRY_EACH_INIT * N_INIT) // analyse prescribed starting points
      {
        delete state;
        delete localBestState;
        delete globalBestState;
        strcpy(datafile, "DATA/");
        strcat(datafile, argv[1]);
        strcat(datafile, "_init");
        strncat(datafile, &initID[(nmodes - 1) % N_INIT], 1);
        strcat(datafile, ".txt");
        ifstream initfile(datafile);
        state = new DPmixGGM(X, L, n, p, 0.1, initfile);
        initfile.close();
        localBestState = new DPmixGGM(state);
        globalBestState = new DPmixGGM(state);
        num_cases += updateAllXis(1, state, bestList);
        L = state->L;
        score = state->plp;
        for (l = 0; l < L; l++) {
          score += state->pll[l];
        }

      } else if (nmodes <= (TRY_EACH_INIT * N_INIT + N_RANDOM_RESTART) &&
                 (N_RANDOM_RESTART > 0)) // analyse renadom starting points
      {
        randomRestart(rand_myInt(MAXL - 1) + 2, state, 0.1);
        initstates[N_INIT - 1 + nmodes - TRY_EACH_INIT * N_INIT] =
            new DPmixGGM(state);
        initstateID[nmodes] = N_INIT - 1 + nmodes - TRY_EACH_INIT * N_INIT;
        num_cases += updateAllXis(1, state, bestList);
        L = state->L;
        score = state->plp;
        for (l = 0; l < L; l++) {
          score += state->pll[l];
        }
      } else if (nmodes <= (TRY_EACH_INIT * N_INIT + N_RANDOM_RESTART +
                            N_MODES_LIST_RESTART)) {
        int maxI;
        Real maxScore = NEG_INF;
        for (i = 0; i < (nmodes - 1); i++) {
          if (modesList->score_list[i] > maxScore) {
            maxScore = modesList->score_list[i];
            maxI = i;
          }
        }

        // state->CopyState(initstates[1]);
        state->CopyState(initstates[initstateID[maxI]]);
        localBestState->CopyState(state);
        globalBestState->CopyState(state);
        num_cases += updateAllXis(1, state, bestList);
        L = state->L;
        score = state->plp;
        for (l = 0; l < L; l++) {
          score += state->pll[l];
        }
      }
#ifndef USE_GPU
      else {
        bestList = new DPmixGGMlist(sizeOfBestList, n, p);
        int maxI;
        Real maxScore = NEG_INF;
        for (i = 0; i < (nmodes - 1); i++) {
          if (modesList->score_list[i] > maxScore) {
            maxScore = modesList->score_list[i];
            maxI = i;
          }
        }
        gsl_rng_set(rnd, seedset[maxI]);

        // state->CopyState(initstates[1]);
        state->CopyState(initstates[initstateID[maxI]]);
        localBestState->CopyState(state);
        globalBestState->CopyState(state);
        num_cases += updateAllXis(1, state, bestList);
        L = state->L;
        score = state->plp;
        for (l = 0; l < L; l++) {
          score += state->pll[l];
        }
      }
#endif
    }
    // SEARCH RESTART
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // SCORE RECORDING
    // ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    L = state->L;
    score = state->plp;
    for (l = 0; l < L; l++) {
      score += state->pll[l];
    }
    num_allModels += num_cases;
    /*
    printf("k=%ld L=%d score=%.4f localBestScore=%.4f globalBestScore=%.4f "
           "nmodes=%d num_cases=%d num_allModels=%ld\n",
           k, state->L, score, localBestScore, globalBestScore, nmodes,
           num_cases, num_allModels);
    */

    if (score > localBestScore) {
      localBestScore = score;
      wastedIterations = 0;
      localJumpCount = 0;
      localBestState->CopyState(state);
    } else {
      wastedIterations++;
    }
    if (score > globalBestScore) {
      globalBestScore = score;
      wastedIterations = 0;
      globalJumpCount = 0;
      globalBestState->CopyState(state);
      featureList->UpdateList(state);
    } else {
      wastedIterations++;
    }
    // SCORE RECORDING
  } // while
  auto now = std::chrono::steady_clock::now();
  auto wall_time =  std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
  printf("final: wall_time=%.4f (s) num_cases=%d num_allModels=%ld\n",
         wall_time * 1e-9, num_cases, num_allModels);

  // writing the lists
  modesList->WriteList(outmodes);
#ifndef USE_GPU
  bestList->WriteList(outbest);
#endif

// cleanups
#ifdef USE_GPU
  hipFree(device.d_in_add);
  hipFree(device.d_in_delete);
  hipFree(device.d_which_add);
  hipFree(device.d_which_delete);
  hipStreamDestroy(device.delete_stream);
  hipStreamDestroy(device.add_stream);
  delete[] device.h_in_delete;
  delete[] device.h_in_add;
  delete[] device.which_delete;
  delete[] device.which_add;

#endif

  outmodes.close();
#ifndef USE_GPU
  outbest.close();
#endif
#endif

  ////////////////////////////////////////////////////////////// END OF SSS
  //////////////////////////////////////////////////////////////////

  // cleanups
  gsl_rng_free(rnd);
  delete[] X;
  gsl_integration_workspace_free(w);
}
