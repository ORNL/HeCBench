#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "kernels.h"

#include "reference.h"

float findBestGraph(sycl::queue &q, float* D_localscore, int* D_resP,
                    float* D_Score, bool *D_parent);

int main(int argc, char** argv) {

  if (argc != 3) {
    printf("Usage: ./%s <path to output file> <repeat>\n", argv[0]);
    return 1;
  }

  // save output in a file
  FILE *fpout = fopen(argv[1], "w");
  if (fpout == NULL) {
    printf("Error: failed to open %s. Exit..\n", argv[1]);
    return -1;
  }

  const int repeat = atoi(argv[2]);

  int i, j, c = 0, tmp, a, b;
  float tmpd;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("NODE_N=%d\nInitialization...\n", NODE_N);

  srand(2);
  initial(); // update sizepernode
  scores = (float*) malloc ((sizepernode / (256 * WORKLOAD) + 1) * sizeof(float));
  parents = (int*) malloc ((sizepernode / (256 * WORKLOAD) + 1) * 4 * sizeof(int));

  Pre_logGamma();

  int *D_data = sycl::malloc_device<int>(NODE_N * DATA_N, q);
  q.memcpy(D_data, data, NODE_N * DATA_N * sizeof(int));

  float *D_LG = sycl::malloc_device<float>(DATA_N + 2, q);
  q.memcpy(D_LG, LG, (DATA_N + 2) * sizeof(float));

  float *D_localscore = sycl::malloc_device<float>(NODE_N * sizepernode, q);

  float *D_Score = sycl::malloc_device<float>(sizepernode / (256 * WORKLOAD) + 1, q);

  bool *D_parent = sycl::malloc_device<bool>(NODE_N, q);

  int *D_resP = sycl::malloc_device<int>((sizepernode / (256 * WORKLOAD) + 1) * 4, q);

  sycl::range<1> gws((sizepernode+255)/256*256);
  sycl::range<1> lws(256);

  const int sizePerNode = sizepernode;  // global variable not allowed by lambda

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    q.memset(D_localscore, 0, NODE_N * sizepernode * sizeof(float));
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class genScore>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        genScoreKernel(sizePerNode, D_localscore, D_data, D_LG, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of genScoreKernel: %f (s)\n", time * 1e-9f / repeat);

  q.memcpy(localscore, D_localscore, NODE_N * sizepernode * sizeof(float)).wait();

  genScore_ref(sizepernode, localscore_ref, data, LG);
  bool ok = true;
  for (int i = 0; i < NODE_N * sizepernode; i++) {
    if (fabsf(localscore[i] - localscore_ref[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("Check genScoreKernel results: %s\n", ok ? "PASS" : "FAIL");

  if (ok) {
    long findBestGraph_time = 0;
    i = 0;
    while (i != ITER) {

      i++;
      score = 0;

      for (a = 0; a < NODE_N; a++) {
        for (j = 0; j < NODE_N; j++) {
          orders[a][j] = preOrders[a][j];
        }
      }

      tmp = rand() % 6;
      for (j = 0; j < tmp; j++)
        genOrders();

      start = std::chrono::steady_clock::now();

      score = findBestGraph(q, D_localscore, D_resP, D_Score, D_parent);

      end = std::chrono::steady_clock::now();
      findBestGraph_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      score_ref = findBestGraph_ref(localscore);
      if (fabsf(score - score_ref) > 1e-3f) {
        ok = false;
        printf("Check scores error: %f != %f (reference)\n", score, score_ref);
        break;
      }

      ConCore();

      // store the top HIGHEST highest orders
      if (c < HIGHEST) {
        tmp = 1;
        for (j = 0; j < c; j++) {
          if (maxScore[j] == preScore) {
            tmp = 0;
          }
        }
        if (tmp != 0) {
          maxScore[c] = preScore;
          for (a = 0; a < NODE_N; a++) {
            for (b = 0; b < NODE_N; b++) {
              bestGraph[c][a][b] = preGraph[a][b];
            }
          }
          c++;
        }

      } else if (c == HIGHEST) {
        sortGraph();
        c++;
      } else {

        tmp = 1;
        for (j = 0; j < HIGHEST; j++) {
          if (maxScore[j] == preScore) {
            tmp = 0;
            break;
          }
        }
        if (tmp != 0 && preScore > maxScore[HIGHEST - 1]) {
          maxScore[HIGHEST - 1] = preScore;
          for (a = 0; a < NODE_N; a++) {
            for (b = 0; b < NODE_N; b++) {
              bestGraph[HIGHEST - 1][a][b] = preGraph[a][b];
            }
          }
          b = HIGHEST - 1;
          for (a = HIGHEST - 2; a >= 0; a--) {
            if (maxScore[b] > maxScore[a]) {
              swap(a, b);
              tmpd = maxScore[a];
              maxScore[a] = maxScore[b];
              maxScore[b] = tmpd;
              b = a;
            }
          }
        }
      }
    } // endwhile

    printf("Find best graph time %lf (s)\n", findBestGraph_time * 1e-9);
  }
  free(LG);
  free(localscore);
  free(localscore_ref);
  free(scores);
  free(parents);
  sycl::free(D_LG, q);
  sycl::free(D_data, q);
  sycl::free(D_localscore, q);
  sycl::free(D_parent, q);
  sycl::free(D_Score, q);
  sycl::free(D_resP, q);

  for(j=0;j<HIGHEST;j++){
    fprintf(fpout,"score:%f\n",maxScore[j]);
    fprintf(fpout,"Best Graph:\n");
    for(int a=0;a<NODE_N;a++){
      for(int b=0;b<NODE_N;b++)
        fprintf(fpout,"%d ",bestGraph[j][a][b]);
      fprintf(fpout,"\n");
    }
    fprintf(fpout,"--------------------------------------------------------------------\n");
  }

  return 0;
}


float findBestGraph(sycl::queue &q, float *D_localscore, int *D_resP,
                    float *D_Score, bool *D_parent) {
  float bestls = -99999999.f;
  int bestparent[5];
  int bestpN, total;
  int node;
  int posN = 0, i, j, parN, tmp;
  float score = 0.f;
  int blocknum;

  for (i = 0; i < NODE_N; i++)
    for (j = 0; j < NODE_N; j++)
      graph[i][j] = 0;

  for (node = 0; node < NODE_N; node++) {

    bestls = -99999999.f;
    posN = 0;

    for (i = 0; i < NODE_N; i++) {
      if (orders[node][i] == 1) {
        posN++;
      }
    }

    total = C(posN, 4) + C(posN, 3) + C(posN, 2) + posN + 1;
    blocknum = total / (256 * WORKLOAD) + 1;

    q.memset(D_resP, 0, blocknum * 4 * sizeof(int));
    q.memset(D_Score, -999999.f, blocknum * sizeof(float));
    q.memcpy(D_parent, orders[node], NODE_N * sizeof(bool));

    const int sizePerNode = sizepernode;
    sycl::range<1> gws(blocknum * 256);
    sycl::range<1> lws(256);

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> sinblock(sycl::range<1>(256), cgh);
      cgh.parallel_for<class compute>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        computeKernel(WORKLOAD, sizePerNode, D_localscore,
          D_parent, node, total, D_Score, D_resP,
          sinblock.get_multi_ptr<sycl::access::decorated::no>().get(), item);
      });
    });

    q.memcpy(parents, D_resP, blocknum * 4 * sizeof(int));
    q.memcpy(scores, D_Score, blocknum * sizeof(float));

    q.wait();

    for (i = 0; i < blocknum; i++) {

      if (scores[i] > bestls) {

        bestls = scores[i];

        parN = 0;
        for (tmp = 0; tmp < 4; tmp++) {
          if (parents[i * 4 + tmp] < 0)
            break;

          bestparent[tmp] = parents[i * 4 + tmp];

          parN++;
        }

        bestpN = parN;
      }
    }

    if (bestls > -99999999.f) {

      for (i = 0; i < bestpN; i++) {
        if (bestparent[i] < node)
          graph[node][bestparent[i] - 1] = 1;
        else
          graph[node][bestparent[i]] = 1;
      }
      score += bestls;
    }
  }

  return score;
}


void sortGraph() {
  float max = -99999999999999.f;
  int maxi, i, j;
  float tmp;

  for (j = 0; j < HIGHEST - 1; j++) {
    max = maxScore[j];
    maxi = j;
    for (i = j + 1; i < HIGHEST; i++) {
      if (maxScore[i] > max) {
        max = maxScore[i];
        maxi = i;
      }
    }

    swap(j, maxi);
    tmp = maxScore[j];
    maxScore[j] = max;
    maxScore[maxi] = tmp;
  }
}

void swap(int a, int b) {
  int i, j;
  bool tmp;

  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++) {

      tmp = bestGraph[a][i][j];
      bestGraph[a][i][j] = bestGraph[b][i][j];
      bestGraph[b][i][j] = tmp;
    }
  }
}

void initial() {
  int i, j, tmp, a, b, r;
  bool tmpd;
  tmp = 1;
  for (i = 1; i <= 4; i++) {
    tmp += C(NODE_N - 1, i);
  }
  sizepernode = tmp;
  tmp *= NODE_N;

  localscore = (float*) malloc(tmp * sizeof(float));
  localscore_ref = (float*) malloc(tmp * sizeof(float));

  for (i = 0; i < tmp; i++)
    localscore_ref[i] = localscore[i] = 0;

  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++)
      orders[i][j] = 0;
  }
  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < i; j++)
      orders[i][j] = 1;
  }
  r = rand() % 10000;
  for (i = 0; i < r; i++) {
    a = rand() % NODE_N;
    b = rand() % NODE_N;
    for (j = 0; j < NODE_N; j++) {
      tmpd = orders[j][a];
      orders[j][a] = orders[j][b];
      orders[j][b] = tmpd;
    }

    for (j = 0; j < NODE_N; j++) {
      tmpd = orders[a][j];
      orders[a][j] = orders[b][j];
      orders[b][j] = tmpd;
    }
  }

  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++) {
      preOrders[i][j] = orders[i][j];
    }
  }
}

// generate ramdom order
int genOrders() {

  int a, b, j;
  bool tmp;
  a = rand() % NODE_N;
  b = rand() % NODE_N;

  for (j = 0; j < NODE_N; j++) {
    tmp = orders[a][j];
    orders[a][j] = orders[b][j];
    orders[b][j] = tmp;
  }
  for (j = 0; j < NODE_N; j++) {
    tmp = orders[j][a];
    orders[j][a] = orders[j][b];
    orders[j][b] = tmp;
  }

  return 1;
}

// decide leave or discard an order
int ConCore() {
  int i, j;
  float tmp;
  tmp = log((rand() % 100000) / 100000.0);
  if (tmp < (score - preScore)) {

    for (i = 0; i < NODE_N; i++) {
      for (j = 0; j < NODE_N; j++) {
        preOrders[i][j] = orders[i][j];
        preGraph[i][j] = graph[i][j];
      }
    }
    preScore = score;

    return 1;
  }

  return 0;
}

void genScore() {
}

void Pre_logGamma() {

  LG = (float*) malloc ((DATA_N + 2) * sizeof(float));

  LG[0] = 0.f; // log(0!) = 0; Nij counts can be zero
  LG[1] = log(1.0);
  float i;
  for (i = 2; i <= DATA_N + 1; i++) {
    LG[(int)i] = LG[(int)i - 1] + log((float)i);
  }
}

void incr(int *bit, int n) {

  bit[n]++;
  if (bit[n] >= 2) {
    bit[n] = 0;
    incr(bit, n + 1);
  }

  return;
}

void incrS(int *bit, int n) {

  bit[n]++;
  if (bit[n] >= STATE_N) {
    bit[n] = 0;
    incr(bit, n + 1);
  }

  return;
}

bool getState(int parN, int *state, int time) {
  int j = 1;

  j = pow(STATE_N, (float)parN) - 1;

  if (time > j)
    return false;

  if (time >= 1)
    incrS(state, 0);

  return true;
}

int findindex(int *arr, int size) { // reminder: arr[0] has to be 0 && size ==
  // array size-1 && index start from 0
  int i, j, index = 0;

  for (i = 1; i < size; i++) {
    index += C(NODE_N - 1, i);
  }

  for (i = 1; i <= size - 1; i++) {
    for (j = arr[i - 1] + 1; j <= arr[i] - 1; j++) {
      index += C(NODE_N - 1 - j, size - i);
    }
  }

  index += arr[size] - arr[size - 1];

  return index;
}

int C(int n, int a) {
  int i, res = 1, atmp = a;

  for (i = 0; i < atmp; i++) {
    res *= n;
    n--;
  }

  for (i = 0; i < atmp; i++) {
    res /= a;
    a--;
  }

  return res;
}
