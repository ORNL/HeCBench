#define SSSMOVES_CPP

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

#ifdef USE_GPU
#ifndef KERNELS
#include "kernels.cpp"
#include <cmath>

#endif
#endif

// Shotgun search all decomposable neighbors of G_l
int updateOneEdgeInOneG(myInt l, myInt nBestLeft, LPGraph G, myInt *xi, State a,
                        List bestlist) {
  // Making a local copy of DPmixGGM class
  int n = a->n;
  int p = a->p;
  Real *X = a->X;
  myInt L = a->L;
  LPGraph *graphlist = a->graphlist;
  Real plp = a->plp;
  Real *pll = a->pll;

  // Other declarations
  int i, j, ii;
  int ee = p * (p - 1) / 2;

  // Variables, follows paper
  Real n0 = N0;
  Real *mu0 = new Real[p];
  for (i = 0; i < p; i++)
    mu0[i] = 0;
  Real *xbar = new Real[p];
  Real *mu_bar = new Real[p];
  Real *D_prior = new Real[p * (p + 1) / 2];
  Real *D_post = new Real[p * (p + 1) / 2];

  // Collect Cluster Information
  myInt n_sub = 0;
  for (i = 0; i < n; i++) {
    if (xi[i] == l)
      n_sub++;
  };
  make_sub_means_and_cov(X, xi, l, p, n, n_sub, xbar, D_post);
  for (i = 0; i < p; i++)
    mu_bar[i] = (n_sub * xbar[i] + n0 * mu0[i]) / (n_sub + n0);
  for (i = 0; i < p * (p + 1) / 2; i++)
    D_prior[i] = 0;
  for (i = 0; i < p; i++)
    D_prior[i * (i + 1) / 2 + i] = 1;
  for (i = 0; i < p; i++)
    D_post[i * (i + 1) / 2 + i] += 1;

  // Factor in the mean for the matrix D_posterior
  for (i = 0; i < p; i++) {
    ii = i * (i + 1) / 2;
    for (j = 0; j <= i; j++) {
      D_post[ii + j] += -(n_sub + n0) * mu_bar[i] * mu_bar[j] +
                        n_sub * xbar[i] * xbar[j] + n0 * mu0[i] * mu0[j];
    }
  }

  // More declarations
  int *which = new int[ee];
  Real *score = new Real[ee + 1];
  int which_ab;
  Real temp_score = j_g_decomposable(G, D_prior, D_post, DELTA0, n_sub, 0);
  Real sumpll = plp;
  for (i = 0; i < L; i++) {
    if (i != l) {
      sumpll += pll[i];
    }
  };
  Real Norm_terms =
      -(Real(n_sub * p) / 2) * log_2_pi + Real(p) / 2 * log(n0 / (n_sub + n0));

  // determine edges for CanAddEdge and CanDeleteEdge
  int n_add = 0;
  int n_delete = 0;
  int d = 0;
  int *a_add = new int[ee];
  int *b_add = new int[ee];
  int *a_delete = new int[ee];
  int *b_delete = new int[ee];
  for (i = 0; i < p - 1; i++) {
    for (j = i + 1; j < p; j++) {
      ii = G->Edge[i][j];
      d += ii;
      if (ii) {
        a_delete[n_delete] = i;
        b_delete[n_delete] = j;
        n_delete++;
      } else {
        a_add[n_add] = i;
        b_add[n_add] = j;
        n_add++;
      }
    }
  }

  // Decomposability check and score computation as in Giudici & Green 1999
#ifndef USE_GPU
  int ndecomp = 0;
  for (i = 0; i < n_delete; i++) {
    which_ab = G->CanDeleteEdge(a_delete[i], b_delete[i]);
    if (which_ab != -1) {
      which[ndecomp] = i;
      score[ndecomp] =
          G->ScoreDeleteEdge(a_delete[i], b_delete[i], which_ab, D_prior,
                             D_post, DELTA0, n_sub, temp_score, d);
      if (bestlist != NULL) {
        graphlist[l]->Edge[a_delete[i]][b_delete[i]] = 0;
        graphlist[l]->Edge[b_delete[i]][a_delete[i]] = 0;
        bestlist->UpdateList(L, xi, graphlist,
                             (sumpll + score[ndecomp] + Norm_terms));
        graphlist[l]->Edge[a_delete[i]][b_delete[i]] = 1;
        graphlist[l]->Edge[b_delete[i]][a_delete[i]] = 1;
      }
      ndecomp++;
    }
  }
  int num_CanDelete = ndecomp;

  for (i = 0; i < n_add; i++) {
    if (G->CanAddEdge(a_add[i], b_add[i])) {
      which[ndecomp] = i;
      score[ndecomp] = G->ScoreAddEdge(a_add[i], b_add[i], D_prior, D_post,
                                       DELTA0, n_sub, temp_score, d);
      if (bestlist != NULL) {
        graphlist[l]->Edge[a_add[i]][b_add[i]] = 1;
        graphlist[l]->Edge[b_add[i]][a_add[i]] = 1;
        bestlist->UpdateList(L, xi, graphlist,
                             (sumpll + score[ndecomp] + Norm_terms));
        graphlist[l]->Edge[a_add[i]][b_add[i]] = 0;
        graphlist[l]->Edge[b_add[i]][a_add[i]] = 0;
      }
      ndecomp++;
    }
  }

#else
  ////////////////////////////////////////////////////////////////////////////////////

  // Variable declarations
  int k, r = 0, n_add_till_now;
  int last;
  size_t size_temp;
  int ndecomp = 0;
  myInt buffsiz;

  // Make CanDeleteEdge buffer and transfer to the GPUs
  device.n_delete = n_delete;
  last = 0;
  device.h_in_delete[last] = p;
  last++;
  device.h_in_delete[last] = G->nCliques;
  last++;
  for (j = 0; j < G->nCliques; j++) {
    device.h_in_delete[last + j] = G->CliquesDimens[j];
  };
  last += G->nCliques;
  for (j = 0; j < G->nCliques; j++) {
    for (k = 0; k < p; k++) {
      device.h_in_delete[last + j * p + k] = G->Cliques[j][k];
    }
  };
  last += p * (G->nCliques);
  device.h_in_delete[last] = device.n_delete;
  last++;
  for (j = 0; j < device.n_delete; j++) {
    device.h_in_delete[last + j] = a_delete[r * (device.n_delete) + j];
  };
  last += device.n_delete;
  for (j = 0; j < device.n_delete; j++) {
    device.h_in_delete[last + j] = b_delete[r * (device.n_delete) + j];
  };
  last += device.n_delete;

  size_temp =
      sizeof(myInt) * (3 + (G->nCliques) * (1 + p) + 2 * device.n_delete);
  device.delete_stream.memcpy(device.d_in_delete, device.h_in_delete, size_temp);

  // Submit CanDeleteEdge
  device.delete_stream.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<myInt, 0> count_acc(cgh);
    sycl::local_accessor<myInt, 0> contain_a_acc(cgh);
    sycl::local_accessor<myInt, 0> contain_b_acc(cgh);
    sycl::local_accessor<myInt, 0> which_ab_acc(cgh);

    short *device_d_in_delete = device.d_in_delete;
    short *device_d_which_delete = device.d_which_delete;

    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(device.n_delete) *
                                       sycl::range<1>(BLOCK_SIZE),
                                       sycl::range<1>(BLOCK_SIZE)),
      [=](sycl::nd_item<1> item) {
      CanDeleteEdge(device_d_in_delete,
                    device_d_which_delete, item,
                    count_acc, contain_a_acc,
                    contain_b_acc, which_ab_acc);
    });
  });

  // Make CanAddEdge buffer and transfer to the GPUs
  buffsiz = p + 2 * (G->nTreeEdges) + BLOCK_SIZE; // Shared memory-size for CanAddEdge
  // cout << "buffsiz = " << buffsiz << endl;
  n_add_till_now = 0;
  device.n_add = (int)n_add;
  last = 0;
  n_add_till_now += device.n_add;
  while (n_add_till_now > n_add) {
    device.n_add--;
    n_add_till_now--;
  }
  // cout << "device.n_add = " << device.n_add << ", "; fflush(stdout);
  for (j = 0; j < p; j++) {
    device.h_in_add[last + j] = G->Labels[j];
  };
  last += p;
  device.h_in_add[last] = G->nSeparators;
  last++;
  for (j = 0; j < G->nSeparators; j++) {
    device.h_in_add[last + j] = G->SeparatorsDimens[j];
  };
  last += G->nSeparators;
  for (j = 0; j < G->nSeparators; j++) {
    for (k = 0; k < p; k++) {
      device.h_in_add[last + j * p + k] = G->Separators[j][k];
    }
  };
  last += p * (G->nSeparators);
  device.h_in_add[last] = G->nTreeEdges;
  last++;
  for (j = 0; j < G->nTreeEdges; j++) {
    device.h_in_add[last + j] = G->TreeEdgeA[j];
  };
  last += G->nTreeEdges;
  for (j = 0; j < G->nTreeEdges; j++) {
    device.h_in_add[last + j] = G->TreeEdgeB[j];
  };
  last += G->nTreeEdges;
  for (j = 0; j < p; j++) {
    for (k = 0; k < p; k++) {
      device.h_in_add[last + j * p + k] = G->Edge[j][k];
    }
  };
  last += p * p;
  device.h_in_add[last] = device.n_add;
  last++;
  for (j = 0; j < device.n_add; j++) {
    device.h_in_add[last + j] = a_add[(n_add_till_now - device.n_add) + j];
  };
  last += device.n_add;
  for (j = 0; j < device.n_add; j++) {
    device.h_in_add[last + j] = b_add[(n_add_till_now - device.n_add) + j];
  };
  last += device.n_add;

  size_temp = sizeof(myInt) * (3 + p + (G->nSeparators) * (1 + p) +
                               2 * (G->nTreeEdges) + p * p + 2 * device.n_add);

  device.add_stream.memcpy(device.d_in_add, device.h_in_add, size_temp);

  device.add_stream.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<myInt, 1> shmem_acc(sycl::range<1>(buffsiz), cgh);
    sycl::local_accessor<myInt, 0> nS_acc(cgh);
    sycl::local_accessor<myInt, 0> pR_acc(cgh);
    sycl::local_accessor<myInt, 0> pT_acc(cgh);
    sycl::local_accessor<myInt, 0> contain_a_acc(cgh);
    sycl::local_accessor<myInt, 0> contain_b_acc(cgh);
    sycl::local_accessor<myInt, 0> aSi_acc(cgh);
    sycl::local_accessor<myInt, 0> bSi_acc(cgh);
    sycl::local_accessor<myInt, 0> common_parent_acc(cgh);
    sycl::local_accessor<myInt, 0> a_acc(cgh);
    sycl::local_accessor<myInt, 0> b_acc(cgh);
    sycl::local_accessor<myBool, 0> flag_acc(cgh);

    short *device_d_in_delete = device.d_in_delete;
    short *device_d_in_add = device.d_in_add;
    short *device_d_which_add = device.d_which_add;

    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(device.n_add * BLOCK_SIZE),
                                       sycl::range<1>(BLOCK_SIZE)),
      [=](sycl::nd_item<1> item) {
      CanAddEdge(device_d_in_delete, device_d_in_add,
                 device_d_which_add, item,
                 shmem_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                 nS_acc, pR_acc, pT_acc, contain_a_acc,
                 contain_b_acc, aSi_acc, bSi_acc,
                 common_parent_acc, a_acc, b_acc,
                 flag_acc);
    });
  });

  // Get CanDeleteEdge results for the GPU
  size_temp = sizeof(myInt) * (device.n_delete);
  device.delete_stream.memcpy(device.which_delete, device.d_which_delete,
                               size_temp).wait();

  // DeleteEdge
  for (k = 0; k < device.n_delete; k++) {
    if (device.which_delete[k] != -1) {
      which[ndecomp] = k;
      score[ndecomp] =
          G->ScoreDeleteEdge(a_delete[k], b_delete[k], device.which_delete[k],
                             D_prior, D_post, DELTA0, n_sub, temp_score, d);
      if (bestlist != NULL) {
        graphlist[l]->Edge[a_delete[k]][b_delete[k]] = 0;
        graphlist[l]->Edge[b_delete[k]][a_delete[k]] = 0;
        bestlist->UpdateList(L, xi, graphlist,
                             (sumpll + score[ndecomp] + Norm_terms));
        graphlist[l]->Edge[a_delete[k]][b_delete[k]] = 1;
        graphlist[l]->Edge[b_delete[k]][a_delete[k]] = 1;
      }
      ndecomp++;
    }
  }
  int num_CanDelete = ndecomp;

  // Get CanAddEdge results from the GPU
  size_temp = sizeof(myInt) * (device.n_add);
  device.add_stream.memcpy(device.which_add, device.d_which_add, size_temp).wait();

  // AddEdge
  n_add_till_now = 0;
  for (k = 0; k < device.n_add; k++) {
    j = n_add_till_now + k;
    if (j >= n_add) {
      break;
    };
    if (device.which_add[k]) {
      which[ndecomp] = j; // cout << j << " ";
      score[ndecomp] = G->ScoreAddEdge(a_add[j], b_add[j], D_prior, D_post,
                                       DELTA0, n_sub, temp_score, d);
      if (bestlist != NULL) {
        graphlist[l]->Edge[a_add[j]][b_add[j]] = 1;
        graphlist[l]->Edge[b_add[j]][a_add[j]] = 1;
        bestlist->UpdateList(L, xi, graphlist,
                             (sumpll + score[ndecomp] + Norm_terms));
        graphlist[l]->Edge[a_add[j]][b_add[j]] = 0;
        graphlist[l]->Edge[b_add[j]][a_add[j]] = 0;
      }
      ndecomp++;
    }
  }
  n_add_till_now += device.n_add;

////////////////////////////////////////////////////////////////////////////////////
#endif

  // ndecomp+1-th graph is true-graph
  score[ndecomp] = temp_score;

  // Samling the new model
  Real maxScore;
  int maxI;
  for (i = 0; i < nBestLeft; i++) {
    maxScore = score[0];
    maxI = 0;
    for (j = 1; j <= ndecomp; j++) {
      if (score[j] > maxScore) {
        maxScore = score[j];
        maxI = j;
      }
    };
    score[maxI] = NEG_INF;
  }
  maxScore = score[0];
  for (i = 1; i <= ndecomp; i++) {
    if (score[i] > maxScore) {
      maxScore = score[i];
    }
  };
  for (i = 0; i <= ndecomp; i++) {
    score[i] -= maxScore;
  }
  Real sumScore = 0;
  for (i = 0; i <= ndecomp; i++) {
    sumScore += exp(score[i]);
  };
  for (i = 0; i <= ndecomp; i++) {
    score[i] = exp(score[i]) / sumScore;
  }
  i = rand_int_weighted(ndecomp + 1, score);

  // Necessary changes in G. which[] has one entry per candidate edge;
  // i == ndecomp keeps the current graph and has no which[] slot.
  if (i < num_CanDelete) {
    int which_change = which[i];
    G->Edge[a_delete[which_change]][b_delete[which_change]] = 0;
    G->Edge[b_delete[which_change]][a_delete[which_change]] = 0;
    if (!G->IsDecomposable()) {
      cout << "Error in CanDeleteEdge." << endl;
      TurnFillInGraph(G);
      G->GenerateAllCliques();
    }
  } else if (i < ndecomp) {
    int which_change = which[i];
    G->Edge[a_add[which_change]][b_add[which_change]] = 1;
    G->Edge[b_add[which_change]][a_add[which_change]] = 1;
    if (!G->IsDecomposable()) {
      cout << "Error in CanAddEdge." << endl;
      TurnFillInGraph(G);
      G->GenerateAllCliques();
    }
  }

  // memory cleanup and exit
  delete[] mu0;
  delete[] xbar;
  delete[] mu_bar;
  delete[] D_prior;
  delete[] D_post;
  delete[] which;
  delete[] score;
  delete[] a_add;
  delete[] b_add;
  delete[] a_delete;
  delete[] b_delete;

  return (ee);
}

long int updateOneEdgeInEveryG(myInt L, myInt *thisl, myInt nBestLeft,
                               LPGraph *graphlist, Real *pll, myInt *thisxi,
                               State a, List bestlist) {
  long int num_cases = 0;
  myInt *xi;
  for (myInt l = 0; l < L; l++) {
    if (thisxi == NULL) {
      xi = a->xi;
    } else {
      xi = thisxi + l * a->n;
    }
    if (thisl == NULL) {
      num_cases +=
          updateOneEdgeInOneG(l, nBestLeft, graphlist[l], xi, a, bestlist);
      if (pll != NULL) {
        pll[l] = a->cluster_k_loglikelihood(l, xi, graphlist[l]);
      }
    } else {
      num_cases += updateOneEdgeInOneG(thisl[l], nBestLeft, graphlist[l], xi, a,
                                       bestlist);
      if (pll != NULL) {
        pll[l] = a->cluster_k_loglikelihood(thisl[l], xi, graphlist[l]);
      }
    }
  }

  return (num_cases);
}

// Shotgun global move for graphs
long int globalJumpOneG(myInt l, myInt size, myInt lookForwardLength,
                        Real sFactor, bool force, State a, List list,
                        List bestlist) {
  myInt i, j;
  long int num_cases = 0;

  // Making a local copy of DMPState class
  myInt n = a->n;
  myInt p = a->p;
  myInt *xi = a->xi;
  myInt L = a->L;
  LPGraph *graphlist = a->graphlist;
  Real plp = a->plp;
  Real *pll = a->pll;
  Real alpha = a->alpha;

  myInt how_many = 0;
  myInt which_ones[n];
  for (i = 0; i < n; i++) {
    if (xi[i] == l) {
      which_ones[how_many] = i;
      how_many++;
    }
  }
  Real sumpll = plp;
  for (i = 0; i < L; i++) {
    if (i != l) {
      sumpll += pll[i];
    }
  }
  LPGraph newgraph[size];
  Real newscore[size + 1];
  Real pll_new[size];
  LPGraph tempgraph;
  myInt thisl[how_many];
  for (i = 0; i < how_many; i++) {
    thisl[i] = l;
  }
  for (i = 0; i < size; i++) {
    newgraph[i] = list->ProposeGraph(how_many, which_ones, sFactor);
  }
  for (j = 0; j < lookForwardLength; j++) {
    num_cases += updateOneEdgeInEveryG(size, thisl, 0, newgraph, NULL, NULL, a,
                                       bestlist);
  }

  for (i = 0; i < size; i++) {
    pll_new[i] = a->cluster_k_loglikelihood(l, xi, newgraph[i]);
    newscore[i] = pll_new[i];
    if (bestlist != NULL) {
      tempgraph = graphlist[l];
      graphlist[l] = newgraph[i];
      bestlist->UpdateList(L, xi, graphlist, (sumpll + newscore[i]));
      graphlist[l] = tempgraph;
    }
  }
  if (!force) {
    newscore[size] = pll[l];
  } else {
    newscore[size] = NEG_INF;
  }

  // SAMPLE THE NEXT MOVE
  Real maxscore = newscore[0];
  for (i = 1; i <= size; i++) {
    if (newscore[i] > maxscore) {
      maxscore = newscore[i];
    }
  }
  Real totalscore = 0;
  for (i = 0; i <= size; i++) {
    newscore[i] = exp(newscore[i] - maxscore);
    totalscore += newscore[i];
  }
  for (i = 0; i <= size; i++) {
    newscore[i] = newscore[i] / totalscore;
  };
  myInt which_change = rand_myInt_weighted(size + 1, newscore);

  if (which_change < size) {
    delete graphlist[l];
    graphlist[l] = newgraph[which_change];
    pll[l] = pll_new[which_change];
  }
  for (i = 0; i < size; i++) {
    if (i != which_change) {
      delete newgraph[i];
    }
  }

  return (num_cases);
}

long int globalJumpAllG(myInt size, bool force, myInt lookForwardLength,
                        Real sFactor, State a, List list, List bestlist) {
  long int num_cases = 0;
  for (myInt l = 0; l < a->L;
       l++) { // cout << "l = " << l << endl; fflush(stdout);
    num_cases += globalJumpOneG(l, size, lookForwardLength, sFactor, force, a,
                                list, bestlist);
  }
  return (num_cases);
}

// ------ Global update the cluster parameter (xi) :: Split-Merge --------------
long int splitMerge(State a, List list, List bestlist, myInt lookForwardLength,
                    Real sFactor, bool force, myInt nSplit, myInt T) {
  // Making a local copy of DMPState class
  myInt n = a->n;
  myInt p = a->p;
  myInt *xi = a->xi;
  myInt L = a->L;
  LPGraph *graphlist = a->graphlist;
  Real plp = a->plp;
  Real *pll = a->pll;
  Real alpha = a->alpha;
  myInt i, j, k, l, m, r, t;
  myInt flag;
  myInt ee = L * nSplit + L * (L - 1) + 1;
  long int num_cases = 0;

  LPGraph *newgraphlist1 = new LPGraph[L * nSplit + L * (L - 1) / 2];
  LPGraph *newgraphlist2 = new LPGraph[L * nSplit + L * (L - 1) / 2];
  LPGraph buffergraph;

  Real *pll_clus_old = new Real[ee];
  Real *pll_clus_new = new Real[ee];
  Real *plp_store = new Real[ee];
  myInt *xi_new = new myInt[n];
  Real sumpll = 0;
  for (l = 0; l < L; l++) {
    sumpll += pll[l];
  };
  Real *score = new Real[ee];

  // split move
  myInt *xi_store = new myInt[nSplit * L * n];
  Real u, v;
  myInt *thisl = new myInt[L * nSplit];
  for (l = 0; l < L; l++) {
    for (r = 0; r < nSplit; r++) {
      thisl[l * nSplit + r] = l;
    }
  }
  myInt *thisL = new myInt[L * nSplit];
  for (l = 0; l < L; l++) {
    for (r = 0; r < nSplit; r++) {
      thisL[l * nSplit + r] = L;
    }
  }
  myInt how_many;
  myInt *which_ones = new myInt[n];

  // partition initialisations at random
  for (l = 0; l < L; l++) {
    for (r = 0; r < nSplit; r++) {
      for (i = 0; i < n; i++) {
        if (xi[i] == l) {
          if (gsl_ran_flat(rnd, 0.0, 1.0) < 0.5) {
            xi_store[(l * nSplit + r) * n + i] = l;
          } else {
            xi_store[(l * nSplit + r) * n + i] = L;
          }
        } else {
          xi_store[(l * nSplit + r) * n + i] = xi[i];
        }
      }
    }
  }

  // graph proposal
  for (l = 0; l < L; l++) {
    if (!force) {
      for (r = 0; r < nSplit; r++) {
        newgraphlist1[l * nSplit + r] = new Graph();
        newgraphlist1[l * nSplit + r]->InitGraph(p);
        newgraphlist1[l * nSplit + r]->CopyGraph(graphlist[l]);
        newgraphlist2[l * nSplit + r] = new Graph();
        newgraphlist2[l * nSplit + r]->InitGraph(p);
        newgraphlist2[l * nSplit + r]->CopyGraph(graphlist[l]);
      }
    }
    if (force) {
      how_many = 0;
      for (i = 0; i < n; i++) {
        if (xi[i] == l) {
          which_ones[how_many] = i;
          how_many++;
        }
      }
      for (r = 0; r < nSplit; r++) {
        newgraphlist1[l * nSplit + r] =
            list->ProposeGraph(how_many, which_ones, sFactor);
        newgraphlist2[l * nSplit + r] =
            list->ProposeGraph(how_many, which_ones, sFactor);
      }
    }
  }

  // Initial lookforward, if no Gibbs steps
  if (T == 0) {
    for (j = 0; j < lookForwardLength; j++) {
      num_cases += updateOneEdgeInEveryG(L * nSplit, thisl, 0, newgraphlist1,
                                         NULL, xi_store, a, (List)NULL);
      num_cases += updateOneEdgeInEveryG(L * nSplit, thisL, 0, newgraphlist2,
                                         NULL, xi_store, a, (List)NULL);
    }
  }

  // RGMS(t)
  for (t = 0; t < T; t++) {
    for (l = 0; l < L; l++) {
      for (r = 0; r < nSplit; r++) {
        for (i = 0; i < n; i++) {
          if (xi[i] != l) {
            continue;
          }

          xi_store[(l * nSplit + r) * n + i] = l;
          u = a->cluster_k_loglikelihood(l, xi_store + (l * nSplit + r) * n,
                                         newgraphlist1[l * nSplit + r]);
          u += a->cluster_k_loglikelihood(L, xi_store + (l * nSplit + r) * n,
                                          newgraphlist2[l * nSplit + r]);
          u += a->partitionlogPrior(L + 1, xi_store + (l * nSplit + r) * n,
                                    alpha);

          xi_store[(l * nSplit + r) * n + i] = L;
          v = a->cluster_k_loglikelihood(l, xi_store + (l * nSplit + r) * n,
                                         newgraphlist1[l * nSplit + r]);
          v += a->cluster_k_loglikelihood(L, xi_store + (l * nSplit + r) * n,
                                          newgraphlist2[l * nSplit + r]);
          v += a->partitionlogPrior(L + 1, xi_store + (l * nSplit + r) * n,
                                    alpha);

          if (u > v) {
            v = exp(v - u);
            u = 1.0;
          } else {
            u = exp(u - v);
            v = 1.0;
          }
          if (gsl_ran_flat(rnd, 0.0, 1.0) < (u / (u + v))) {
            xi_store[(l * nSplit + r) * n + i] = l;
          } else {
            xi_store[(l * nSplit + r) * n + i] = L;
          }
        }
      }
    }

    for (j = 0; j < lookForwardLength; j++) {
      num_cases += updateOneEdgeInEveryG(L * nSplit, thisl, 0, newgraphlist1,
                                         NULL, xi_store, a, (List)NULL);
      num_cases += updateOneEdgeInEveryG(L * nSplit, thisL, 0, newgraphlist2,
                                         NULL, xi_store, a, (List)NULL);
    }
  }

  for (l = 0; l < L; l++) {
    for (r = 0; r < nSplit; r++) {
      pll_clus_old[l * nSplit + r] = a->cluster_k_loglikelihood(
          l, xi_store + (l * nSplit + r) * n, newgraphlist1[l * nSplit + r]);
      pll_clus_new[l * nSplit + r] = a->cluster_k_loglikelihood(
          L, xi_store + (l * nSplit + r) * n, newgraphlist2[l * nSplit + r]);
      plp_store[l * nSplit + r] =
          a->partitionlogPrior(L + 1, xi_store + (l * nSplit + r) * n, alpha);
      score[l * nSplit + r] = sumpll - pll[l] + pll_clus_old[l * nSplit + r] +
                              pll_clus_new[l * nSplit + r] +
                              plp_store[l * nSplit + r];

      if (bestlist != NULL) {
        LPGraph graphlist_new[L + 1];
        for (i = 0; i < L; i++) {
          graphlist_new[i] = graphlist[i];
        }
        graphlist_new[l] = newgraphlist1[l * nSplit + r];
        graphlist_new[L] = newgraphlist2[l * nSplit + r];
        bestlist->UpdateList(L + 1, xi_store + (l * nSplit + r) * n,
                             graphlist_new, score[l * nSplit + r]);
      }
    }
  }

  // All possible merge-moves -- loop through (i,j) pairs, i<j and merge them.
  // Try both graphs to see which one is better
  myInt start = L * nSplit - 1;
  for (i = 0; i < L - 1; i++) {
    for (j = i + 1; j < L; j++) {
      start++;
      for (k = 0; k < n; k++) {
        if (xi[k] == j) {
          xi_new[k] = i;
        } else {
          xi_new[k] = xi[k];
        }
      }

      newgraphlist1[start] = new Graph();
      newgraphlist1[start]->InitGraph(p);
      newgraphlist1[start]->CopyGraph(graphlist[i]);
      for (k = 0; k < lookForwardLength; k++) {
        num_cases += updateOneEdgeInOneG(i, 0, newgraphlist1[start], xi_new, a,
                                         (List)NULL);
      }
      pll_clus_old[start] =
          a->cluster_k_loglikelihood(i, xi_new, newgraphlist1[start]);

      newgraphlist2[start] = new Graph();
      newgraphlist2[start]->InitGraph(p);
      newgraphlist2[start]->CopyGraph(graphlist[j]);
      for (k = 0; k < lookForwardLength; k++) {
        num_cases += updateOneEdgeInOneG(i, 0, newgraphlist2[start], xi_new, a,
                                         (List)NULL);
      }
      pll_clus_new[start] =
          a->cluster_k_loglikelihood(i, xi_new, newgraphlist2[start]);

      plp_store[start] = a->partitionlogPrior(L - 1, xi_new, alpha);

      score[start] =
          sumpll - pll[i] - pll[j] + pll_clus_old[start] + plp_store[start];
      score[L * (L - 1) / 2 + start] =
          sumpll - pll[i] - pll[j] + pll_clus_new[start] + plp_store[start];

      if (bestlist != NULL) {
        LPGraph graphlist_new[L - 1];
        for (l = 0; l < j - 1; l++) {
          graphlist_new[l] = graphlist[l];
        };
        for (l = j; l < L - 1; l++) {
          graphlist_new[l] = graphlist[l + 1];
        }
        graphlist_new[i] = newgraphlist1[start];
        bestlist->UpdateList(L - 1, xi_new, graphlist_new, score[start]);
        graphlist_new[i] = newgraphlist2[start];
        bestlist->UpdateList(L - 1, xi_new, graphlist_new,
                             score[L * (L - 1) / 2 + start]);
      }
    }
  }

  if (!force) {
    score[ee - 1] = sumpll + plp;
  } else {
    score[ee - 1] = NEG_INF;
  }

  Real maxscore = score[0];
  Real totalscore = 0;
  for (i = 1; i < ee; i++) {
    if (score[i] > maxscore)
      maxscore = score[i];
  };
  for (i = 0; i < ee; i++) {
    score[i] = exp(score[i] - maxscore);
    totalscore += score[i];
  }
  for (i = 0; i < ee; i++) {
    score[i] = score[i] / totalscore;
  };
  myInt which_change = rand_myInt_weighted(ee, score);

  // saving the new model
  Real *pll_new;
  LPGraph *graphlist_new;
  if (which_change < L * nSplit) // split
  {
    for (j = 0; j < n; j++) {
      xi[j] = xi_store[n * which_change + j];
    }

    l = which_change / nSplit;

    // Check if both l & L are present
    bool flagl = 0;
    for (i = 0; i < n; i++) {
      if (xi[i] == l) {
        flagl = 1;
        break;
      }
    }
    bool flagL = 0;
    for (i = 0; i < n; i++) {
      if (xi[i] == L) {
        flagL = 1;
        break;
      }
    }

    if (flagl && flagL) {
      pll_new = new Real[L + 1];
      for (i = 0; i < L; i++) {
        pll_new[i] = pll[i];
      }
      pll_new[l] = pll_clus_old[which_change];
      pll_new[L] = pll_clus_new[which_change];
      delete[] pll;
      a->pll = pll_new;
      pll = a->pll;

      delete graphlist[l];
      graphlist_new = new LPGraph[L + 1];
      for (i = 0; i < L; i++) {
        graphlist_new[i] = graphlist[i];
      }
      graphlist_new[l] = newgraphlist1[which_change];
      graphlist_new[L] = newgraphlist2[which_change];
      delete[] graphlist;
      a->graphlist = graphlist_new;
      graphlist = a->graphlist;
      for (i = 0; i < L * nSplit + L * (L - 1) / 2; i++) {
        if (i != which_change) {
          delete newgraphlist1[i];
          delete newgraphlist2[i];
        }
      }

      L++;
      a->L = L;
    } else if (flagl) {
      pll[l] = pll_clus_old[which_change];
      delete graphlist[l];
      graphlist[l] = newgraphlist1[which_change];
      for (i = 0; i < L * nSplit + L * (L - 1) / 2; i++) {
        if (i != which_change) {
          delete newgraphlist1[i];
        }
        delete newgraphlist2[i];
      }
    } else {
      for (j = 0; j < n; j++) {
        if (xi[j] == L) {
          xi[j] = l;
        }
      }
      pll[l] = pll_clus_new[which_change];
      delete graphlist[l];
      graphlist[l] = newgraphlist2[which_change];
      for (i = 0; i < L * nSplit + L * (L - 1) / 2; i++) {
        if (i != which_change) {
          delete newgraphlist2[i];
        }
        delete newgraphlist1[i];
      }
    }

    plp = plp_store[which_change];
    a->plp = plp;

  } else if (which_change <
             (L * nSplit +
              L * (L - 1) / 2)) // merge, graph = graph of cluster i
  {
    flag = 0;
    start = L * nSplit - 1;
    for (i = 0; i < L - 1; i++) {
      for (j = i + 1; j < L; j++) {
        start++;
        if (start == which_change) {
          flag = 1;
          break;
        }
      };
      if (flag)
        break;
    }
    if ((i == L - 1) && (j == L)) {
      i--;
      j--;
    }

    for (k = 0; k < n; k++) {
      if (xi[k] == j) {
        xi[k] = i;
      } else if (xi[k] > j) {
        xi[k] = xi[k] - 1;
      }
    }

    pll_new = new Real[L - 1];
    for (l = 0; l < j; l++)
      pll_new[l] = pll[l];
    for (l = j + 1; l < L; l++)
      pll_new[l - 1] = pll[l];
    pll_new[i] = pll_clus_old[which_change];
    a->pll = pll_new;
    delete[] pll;
    pll = a->pll;
    plp = plp_store[which_change];
    a->plp = plp;

    graphlist_new = new LPGraph[L - 1];
    delete graphlist[i];
    delete graphlist[j];
    for (l = 0; l <= j - 1; l++) {
      graphlist_new[l] = graphlist[l];
    };
    for (l = j + 1; l < L; l++) {
      graphlist_new[l - 1] = graphlist[l];
    }
    graphlist_new[i] = newgraphlist1[which_change];
    a->graphlist = graphlist_new;
    delete[] graphlist;
    graphlist = a->graphlist;
    for (i = 0; i < L * nSplit + L * (L - 1) / 2; i++) {
      if (i != which_change) {
        delete newgraphlist1[i];
      };
      delete newgraphlist2[i];
    }

    L--;
    a->L = L;
  } else if (which_change <
             L * nSplit + L * (L - 1)) // merge, graph = graph of cluster j
  {
    flag = 0;
    start = L * nSplit + L * (L - 1) / 2 - 1;
    for (i = 0; i < L - 1; i++) {
      for (j = i + 1; j < L; j++) {
        start++;
        if (start == which_change) {
          flag = 1;
          break;
        }
      }
      if (flag)
        break;
    }
    if ((i == L - 1) && (j == L)) {
      i--;
      j--;
    }
    for (k = 0; k < n; k++) {
      if (xi[k] == j) {
        xi[k] = i;
      } else if (xi[k] > j) {
        xi[k] = xi[k] - 1;
      }
    }

    pll_new = new Real[L - 1];
    for (l = 0; l < j; l++)
      pll_new[l] = pll[l];
    for (l = j + 1; l < L; l++)
      pll_new[l - 1] = pll[l];
    pll_new[i] = pll_clus_new[which_change - L * (L - 1) / 2];
    a->pll = pll_new;
    delete[] pll;
    pll = a->pll;

    plp = plp_store[which_change - L * (L - 1) / 2];
    a->plp = plp;

    graphlist_new = new LPGraph[L - 1];
    delete graphlist[i];
    delete graphlist[j];
    for (l = 0; l <= j - 1; l++) {
      graphlist_new[l] = graphlist[l];
    };
    graphlist_new[i] = graphlist[j];
    for (l = j + 1; l < L; l++) {
      graphlist_new[l - 1] = graphlist[l];
    };
    graphlist_new[i] = newgraphlist2[which_change - L * (L - 1) / 2];
    a->graphlist = graphlist_new;
    delete[] graphlist;
    graphlist = a->graphlist;
    for (i = 0; i < L * nSplit + L * (L - 1) / 2; i++) {
      if (i != (which_change - L * (L - 1) / 2)) {
        delete newgraphlist2[i];
      };
      delete newgraphlist1[i];
    }

    L--;
    a->L = L;
  } else {
    for (i = 0; i < L * nSplit + L * (L - 1) / 2; i++) {
      delete newgraphlist1[i];
      delete newgraphlist2[i];
    }
  }

  delete[] score;
  delete[] xi_new;
  delete[] xi_store;
  delete[] pll_clus_old;
  delete[] pll_clus_new;
  delete[] plp_store;
  delete[] newgraphlist1;
  delete[] newgraphlist2;
  delete[] thisl;
  delete[] thisL;
  delete[] which_ones;
  return (num_cases);
}

// ------ Global update the cluster parameter (xi) :: Split-Merge --------------
int Merge(State a, List bestlist, myInt lookForwardLength, bool force) {
  // Making a local copy of DMPState class
  myInt n = a->n;
  myInt p = a->p;
  myInt *xi = a->xi;
  myInt L = a->L;
  LPGraph *graphlist = a->graphlist;
  Real plp = a->plp;
  Real *pll = a->pll;
  Real alpha = a->alpha;
  myInt i, j, k, l, m, r, t;
  myInt flag;
  long int num_cases = 0;

  LPGraph *newgraphlist1 = new LPGraph[L * (L - 1) / 2];
  LPGraph *newgraphlist2 = new LPGraph[L * (L - 1) / 2];

  Real *pll_clus_old = new Real[L * (L - 1)];
  Real *pll_clus_new = new Real[L * (L - 1)];
  Real *plp_store = new Real[L * (L - 1)];
  myInt *xi_new = new myInt[n];
  Real sumpll = 0;
  for (l = 0; l < L; l++) {
    sumpll += pll[l];
  };
  Real *score = new Real[L * (L - 1) + 1];

  // All possible merge-moves -- loop through (i,j) pairs, i<j and merge them.
  // Try both graphs to see which one is better
  myInt start = -1;
  for (i = 0; i < L - 1; i++) {
    for (j = i + 1; j < L; j++) {
      start++;
      for (k = 0; k < n; k++) {
        if (xi[k] == j) {
          xi_new[k] = i;
        } else {
          xi_new[k] = xi[k];
        }
      }

      newgraphlist1[start] = new Graph();
      newgraphlist1[start]->InitGraph(p);
      newgraphlist1[start]->CopyGraph(graphlist[i]);
      for (k = 0; k < lookForwardLength; k++) {
        num_cases += updateOneEdgeInOneG(i, 0, newgraphlist1[start], xi_new, a,
                                         (List)NULL);
      }
      pll_clus_old[start] =
          a->cluster_k_loglikelihood(i, xi_new, newgraphlist1[start]);

      newgraphlist2[start] = new Graph();
      newgraphlist2[start]->InitGraph(p);
      newgraphlist2[start]->CopyGraph(graphlist[j]);
      for (k = 0; k < lookForwardLength; k++) {
        num_cases += updateOneEdgeInOneG(i, 0, newgraphlist2[start], xi_new, a,
                                         (List)NULL);
      }
      pll_clus_new[start] =
          a->cluster_k_loglikelihood(i, xi_new, newgraphlist2[start]);

      plp_store[start] = a->partitionlogPrior(L - 1, xi_new, alpha);

      score[start] =
          sumpll - pll[i] - pll[j] + pll_clus_old[start] + plp_store[start];
      score[L * (L - 1) / 2 + start] =
          sumpll - pll[i] - pll[j] + pll_clus_new[start] + plp_store[start];

      if (bestlist != NULL) {
        LPGraph graphlist_new[L - 1];
        for (l = 0; l < j - 1; l++) {
          graphlist_new[l] = graphlist[l];
        };
        for (l = j; l < L - 1; l++) {
          graphlist_new[l] = graphlist[l + 1];
        }
        graphlist_new[i] = newgraphlist1[start];
        bestlist->UpdateList(L - 1, xi_new, graphlist_new, score[start]);
        graphlist_new[i] = newgraphlist2[start];
        bestlist->UpdateList(L - 1, xi_new, graphlist_new,
                             score[L * (L - 1) / 2 + start]);
      }
    }
  }

  if (!force) {
    score[L * (L - 1)] = sumpll + plp;
  } else {
    score[L * (L - 1)] = NEG_INF;
  }

  Real maxscore = score[0];
  for (i = 1; i <= L * (L - 1); i++) {
    if (score[i] > maxscore)
      maxscore = score[i];
  }
  Real totalscore = 0;
  for (i = 0; i <= L * (L - 1); i++) {
    score[i] = exp(score[i] - maxscore);
    totalscore += score[i];
  }
  for (i = 0; i <= L * (L - 1); i++) {
    score[i] = score[i] / totalscore;
  };
  myInt which_change = rand_myInt_weighted(L * (L - 1) + 1, score);

  // saving the new model
  Real *pll_new;
  LPGraph *graphlist_new;

  if (which_change < L * (L - 1) / 2) // merge, graph = graph of cluster i
  {
    flag = 0;
    start = -1;
    for (i = 0; i < L - 1; i++) {
      for (j = i + 1; j < L; j++) {
        start++;
        if (start == which_change) {
          flag = 1;
          break;
        }
      };
      if (flag)
        break;
    }
    if ((i == L - 1) && (j == L)) {
      i--;
      j--;
    }

    for (k = 0; k < n; k++) {
      if (xi[k] == j) {
        xi[k] = i;
      } else if (xi[k] > j) {
        xi[k] = xi[k] - 1;
      }
    }

    pll_new = new Real[L - 1];
    for (l = 0; l < j; l++)
      pll_new[l] = pll[l];
    for (l = j + 1; l < L; l++)
      pll_new[l - 1] = pll[l];
    pll_new[i] = pll_clus_old[which_change];
    a->pll = pll_new;
    delete[] pll;
    pll = a->pll;
    plp = plp_store[which_change];
    a->plp = plp;

    graphlist_new = new LPGraph[L - 1];
    delete graphlist[i];
    delete graphlist[j];
    for (l = 0; l <= j - 1; l++) {
      graphlist_new[l] = graphlist[l];
    };
    for (l = j + 1; l < L; l++) {
      graphlist_new[l - 1] = graphlist[l];
    }
    graphlist_new[i] = newgraphlist1[which_change];
    a->graphlist = graphlist_new;
    delete[] graphlist;
    graphlist = a->graphlist;
    for (i = 0; i < L * (L - 1) / 2; i++) {
      if (i != which_change) {
        delete newgraphlist1[i];
      };
      delete newgraphlist2[i];
    }

    L--;
    a->L = L;
  } else if (which_change < L * (L - 1)) // merge, graph = graph of cluster j
  {
    flag = 0;
    start = L * (L - 1) / 2 - 1;
    for (i = 0; i < L - 1; i++) {
      for (j = i + 1; j < L; j++) {
        start++;
        if (start == which_change) {
          flag = 1;
          break;
        }
      }
      if (flag)
        break;
    }
    if ((i == L - 1) && (j == L)) {
      i--;
      j--;
    }

    for (k = 0; k < n; k++) {
      if (xi[k] == j) {
        xi[k] = i;
      } else if (xi[k] > j) {
        xi[k] = xi[k] - 1;
      }
    }

    pll_new = new Real[L - 1];
    for (l = 0; l < j; l++)
      pll_new[l] = pll[l];
    for (l = j + 1; l < L; l++)
      pll_new[l - 1] = pll[l];
    pll_new[i] = pll_clus_new[which_change - L * (L - 1) / 2];
    a->pll = pll_new;
    delete[] pll;
    pll = a->pll;

    plp = plp_store[which_change - L * (L - 1) / 2];
    a->plp = plp;

    graphlist_new = new LPGraph[L - 1];
    delete graphlist[i];
    delete graphlist[j];
    for (l = 0; l <= j - 1; l++) {
      graphlist_new[l] = graphlist[l];
    };
    graphlist_new[i] = graphlist[j];
    for (l = j + 1; l < L; l++) {
      graphlist_new[l - 1] = graphlist[l];
    };
    graphlist_new[i] = newgraphlist2[which_change - L * (L - 1) / 2];
    a->graphlist = graphlist_new;
    delete[] graphlist;
    graphlist = a->graphlist;
    for (i = 0; i < L * (L - 1) / 2; i++) {
      if (i != (which_change - L * (L - 1) / 2)) {
        delete newgraphlist2[i];
      };
      delete newgraphlist1[i];
    }

    L--;
    a->L = L;
  } else {
    for (i = 0; i < L * (L - 1) / 2; i++) {
      delete newgraphlist1[i];
      delete newgraphlist2[i];
    }
  }

  delete[] score;
  delete[] xi_new;
  delete[] pll_clus_old;
  delete[] pll_clus_new;
  delete[] plp_store;
  delete[] newgraphlist1;
  delete[] newgraphlist2;
  return (num_cases);
}

//------ Update multiple xi at the same time : to be partly parallelized later
//--------------
int updateManyXiInOneScan(myInt NN_xi, myInt *which_xi, State a,
                          List bestlist) {
  // Making a local copy of DPmixGGM class
  myInt n = a->n;
  myInt p = a->p;
  myInt *xi = a->xi;
  myInt L = a->L;
  LPGraph *graphlist = a->graphlist;
  Real plp = a->plp;
  Real *pll = a->pll;
  Real alpha = a->alpha;

  // other declarations and initialisations
  myInt i, j, k, l;
  myInt temp;
  myInt *allClus = new myInt[2 * NN_xi];

  myInt num_cases = 1;
  for (i = 0; i < NN_xi; i++) {
    num_cases *= L;
  }

  // Setting cluster index for observations other than those in 'which_xi'
  myInt *xi_old = new myInt[NN_xi];
  for (j = 0; j < NN_xi; j++) {
    xi_old[j] = xi[which_xi[j]];
  }
  Real *qs = new Real[num_cases];
  Real sumpll = 0;
  for (l = 0; l < L; l++) {
    sumpll += pll[l];
  }

  // adding a graph for a new cluster
  LPGraph *graphlist_new;
  Real *pll_new;

  myInt tot, flag;
  Real *pll_new_store = new Real[num_cases * L];
  Real *plp_store = new Real[num_cases];
  myInt *xi_new = new myInt[n];
  for (j = 0; j < n; j++) {
    xi_new[j] = xi[j];
  };
  Real qs_temp;

  for (i = 0; i < num_cases * L; i++) {
    pll_new_store[i] = -1.0;
  }

  for (i = 0; i < num_cases; i++) { // first, complete the new cluster proposal
    tot = i;
    for (j = 0; j < NN_xi; j++) {
      xi_new[which_xi[j]] = tot % L;
      tot /= L;
    }

    qs_temp = sumpll;
    for (j = 0; j < NN_xi; j++) {
      allClus[j] = xi_new[which_xi[j]];
    };
    for (j = 0; j < NN_xi; j++) {
      allClus[NN_xi + j] = xi_old[j];
    }
    for (j = 0; j < 2 * NN_xi; j++) {
      l = allClus[j];
      flag = 1;
      for (k = 0; k < j; k++) {
        if (l == allClus[k]) {
          flag = 0;
          break;
        }
      }

      if (flag) // computes only once for each cluster index
      {
        pll_new_store[i * L + l] =
            a->cluster_k_loglikelihood(l, xi_new, graphlist[l]);
        qs_temp = qs_temp - pll[l] + pll_new_store[i * L + l];
      } else {
        pll_new_store[i * L + l] = pll_new_store[i * L + allClus[k]];
      }
    }
    plp_store[i] = a->partitionlogPrior(
        L, xi_new, alpha); // if effective number of clusters is smaller, it's
                           // taken care of
    qs[i] = qs_temp + plp_store[i];

    if (bestlist != NULL) {
      bestlist->UpdateList(L, xi_new, graphlist, qs[i]);
    }
  }
  delete[] xi_new;

  //------------ Now that we've scored everything, choose a model ----------
  Real maxq = qs[0];
  for (i = 1; i < num_cases; i++) {
    if (qs[i] > maxq) {
      maxq = qs[i];
    }
  };
  for (i = 0; i < num_cases; i++) {
    qs[i] -= maxq;
  }
  Real sumq = 0;
  for (i = 0; i < num_cases; i++) {
    sumq += exp(qs[i]);
  };
  for (i = 0; i < num_cases; i++) {
    qs[i] = exp(qs[i]) / sumq;
  }
  k = rand_int_weighted(num_cases, qs);

  //------ Put observation in new cluster and make sure things are
  //consistent----------
  tot = k;
  for (j = 0; j < NN_xi; j++) {
    xi[which_xi[j]] = tot % L;
    tot /= L;
  }
  for (j = 0; j < NN_xi; j++) {
    pll[xi_old[j]] = pll_new_store[k * L + xi_old[j]];
  }
  for (j = 0; j < NN_xi; j++) {
    pll[xi[which_xi[j]]] = pll_new_store[k * L + xi[which_xi[j]]];
  }
  plp = plp_store[k];
  a->plp = plp;

  // Drop redundant clusters. allClus only holds the 2*NN_xi labels
  // scored above; empty clusters can be as many as L.
  myInt count = 0;
  myInt *emptyClus = new myInt[L];

  // find redundant clusters
  for (l = 0; l < L; l++) {
    flag = 1;
    for (i = 0; i < n; i++) {
      if (xi[i] == l) {
        flag = 0;
        break;
      }
    }
    if (flag) {
      emptyClus[count] = l;
      count++;
    }
  }

  // reindexing. Compare against the original id: an in-place decrement
  // makes a later empty cluster look no smaller than the updated id.
  for (i = 0; i < n; i++) {
    myInt xi_orig = xi[i];
    for (j = 0; j < count; j++) {
      if (xi_orig > emptyClus[j]) {
        xi[i] = xi[i] - 1;
      }
    }
  }

  // resizing graphlist and pll-list
  graphlist_new = new LPGraph[L - count];
  pll_new = new Real[L - count];
  for (j = 0; j < count; j++) {
    delete graphlist[emptyClus[j]];
  }
  for (l = 0; l < L; l++) {
    k = 0;
    flag = 0;
    for (j = 0; j < count; j++) {
      if (l == emptyClus[j]) {
        flag = 1;
        break;
      } else if (l > emptyClus[j])
        k++;
    };
    if (flag)
      continue;
    graphlist_new[l - k] = graphlist[l];
    pll_new[l - k] = pll[l];
  }
  a->graphlist = graphlist_new;
  delete[] graphlist;
  graphlist = a->graphlist;
  a->pll = pll_new;
  delete[] pll;
  pll = a->pll;

  // finally redefine number of clusters
  L = L - count;
  a->L = L;

  // cleanup and return
  delete[] pll_new_store;
  delete[] plp_store;
  delete[] xi_old;
  delete[] qs;
  delete[] allClus;
  delete[] emptyClus;
  return (num_cases);
}

int updateAllXis(myInt chunkSize, State a, List bestlist) // random_order ??
{
  myInt i, j;
  myInt current_xi = 0;
  long int num_cases = 0;
  myInt *which_xi = new myInt[chunkSize];

  for (i = 0; i < ceil(Real(a->n) / Real(chunkSize)); i++) {
    for (j = 0; j < chunkSize; j++) {
      which_xi[j] = current_xi % a->n;
      current_xi++;
    }
    num_cases += updateManyXiInOneScan(chunkSize, which_xi, a, bestlist);
  }

  delete[] which_xi;
  return (num_cases);
}

///////////////////////////// RESAMPLING MOVES ///////////////////////////

void resampleOneG(myInt l, State a, List featurelist) {
  myInt n = a->n;
  myInt p = a->p;
  myInt *xi = a->xi;
  LPGraph *graphlist = a->graphlist;
  Real *pll = a->pll;
  myInt M = featurelist->M;
  myInt *L_list = featurelist->L_list;
  myInt *edge_list = featurelist->edge_list;

  myInt i, j, q, r;
  myInt ee = p * (p - 1) / 2;

  r = 1;
  for (i = 1; i < M; i++) {
    if (L_list[i] == -1) {
      break;
    };
    r++;
  };
  myInt t = rand_myInt(r); // randomly choose one of the list models

  // copy stored values
  for (i = 0; i < n; i++) {
    if (xi[i] == l) {
      break;
    }
  }

  j = 0;
  for (q = 0; q < p - 1; q++) {
    graphlist[l]->Edge[q][q] = 0;
    for (r = q + 1; r < p; r++) {
      graphlist[l]->Edge[q][r] = edge_list[t * n * ee + i * ee + j];
      graphlist[l]->Edge[r][q] = graphlist[l]->Edge[q][r];
      j++;
    }
  }
  graphlist[l]->Edge[p - 1][p - 1] = 0;
  graphlist[l]->GenerateAllCliques();
  a->pll[l] = a->cluster_k_loglikelihood(l, xi, graphlist[l]);
}

void resampleAllGindividually(State a, List featurelist) {
  for (myInt l = 0; l < (a->L); l++) {
    resampleOneG(l, a, featurelist);
  }
}

void resampleAllG(State a, List featurelist) {
  // Making a local copy of DMPState class
  myInt n = a->n;
  myInt p = a->p;
  myInt *xi = a->xi;
  myInt L = a->L;
  LPGraph *graphlist = a->graphlist;
  Real *pll = a->pll;
  myInt M = featurelist->M;
  myInt *L_list = featurelist->L_list;
  myInt *edge_list = featurelist->edge_list;

  myInt i, j, l, q, r;
  myInt ee = p * (p - 1) / 2;

  r = 1;
  for (i = 1; i < M; i++) {
    if (L_list[i] == -1) {
      break;
    };
    r++;
  };
  myInt t = rand_myInt(r); // randomly choose one of the list models

  // copy stored values
  for (l = 0; l < L; l++) {
    for (i = 0; i < n; i++) {
      if (xi[i] == l) {
        break;
      }
    }

    j = 0;
    for (q = 0; q < p - 1; q++) {
      graphlist[l]->Edge[q][q] = 0;
      for (r = q + 1; r < p; r++) {
        graphlist[l]->Edge[q][r] = edge_list[t * n * ee + i * ee + j];
        graphlist[l]->Edge[r][q] = graphlist[l]->Edge[q][r];
        j++;
      }
    }
    graphlist[l]->Edge[p - 1][p - 1] = 0;
    graphlist[l]->GenerateAllCliques();
  }

  for (l = 0; l < L; l++) {
    a->pll[l] = a->cluster_k_loglikelihood(l, xi, graphlist[l]);
  }
}

void resampleState(State a, List featurelist) {
  // Making a local copy of DMPState class
  myInt n = a->n;
  myInt p = a->p;
  myInt *xi = a->xi;
  myInt L = a->L;
  LPGraph *graphlist = a->graphlist;
  Real plp = a->plp;
  Real *pll = a->pll;
  Real alpha = a->alpha;

  myInt M = featurelist->M;
  myInt *L_list = featurelist->L_list;
  myInt *xi_list = featurelist->xi_list;
  myInt *edge_list = featurelist->edge_list;

  myInt i, j, l, q, r;
  myInt ee = p * (p - 1) / 2;

  // deleting existing DPmixGGM model memory
  for (l = 0; l < L; l++) {
    delete graphlist[l];
  };
  delete[] graphlist;
  delete[] pll;

  r = 1;
  for (i = 1; i < M; i++) {
    if (L_list[i] == -1) {
      break;
    };
    r++;
  };
  myInt t = rand_myInt(r); // randomly choose one of the list models

  L = L_list[t];
  a->L = L;

  // assign new DPmixGGM model memory
  a->graphlist = new LPGraph[L];
  graphlist = a->graphlist;
  for (l = 0; l < L; l++) {
    graphlist[l] = new Graph();
    graphlist[l]->InitGraph(p);
  }
  a->pll = new Real[L];
  pll = a->pll;

  // copy stored values
  for (i = 0; i < n; i++) {
    xi[i] = xi_list[t * n + i];
  }
  for (l = 0; l < L; l++) {
    for (i = 0; i < n; i++) {
      if (xi[i] == l) {
        break;
      }
    }

    j = 0;
    for (q = 0; q < p - 1; q++) {
      graphlist[l]->Edge[q][q] = 0;
      for (r = q + 1; r < p; r++) {
        graphlist[l]->Edge[q][r] = edge_list[t * n * ee + i * ee + j];
        graphlist[l]->Edge[r][q] = graphlist[l]->Edge[q][r];
        j++;
      }
    }
    graphlist[l]->Edge[p - 1][p - 1] = 0;
    graphlist[l]->GenerateAllCliques();
  }

  for (l = 0; l < L; l++) {
    a->pll[l] = a->cluster_k_loglikelihood(l, xi, graphlist[l]);
  }
  a->plp = a->partitionlogPrior(L, xi, alpha);
}

///////////////////////////// RESTARTING MOVES ///////////////////////////

void randomRestart(myInt L, State a, Real edgeInclusionProb) {
  myInt n = a->n;
  myInt p = a->p;
  myInt *xi = a->xi;
  myInt oldL = a->L;
  LPGraph *graphlist = a->graphlist;
  Real plp = a->plp;
  Real *pll = a->pll;
  Real alpha = a->alpha;

  myInt l;
  for (l = 0; l < oldL; l++) {
    delete graphlist[l];
  };
  delete[] graphlist;
  delete[] pll;
  a->graphlist = new LPGraph[L];
  graphlist = a->graphlist;
  for (l = 0; l < L; l++) {
    graphlist[l] = new Graph();
    graphlist[l]->InitGraph(p);
  }
  a->pll = new Real[L];
  pll = a->pll;
  a->L = L;

  a->RandomStartAllXi(L);
  a->RandomStartAllG(L, edgeInclusionProb);
  plp = a->partitionlogPrior(L, xi, alpha);
  for (l = 0; l < L; l++) {
    pll[l] = a->cluster_k_loglikelihood(l, xi, graphlist[l]);
  }
}
