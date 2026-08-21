#include <cmath>
#define DPMIXGGM_CPP
#ifndef GRAPH_CPP
#include "graph.cpp"
#endif
#ifndef GWISH_CPP
#include "gwish.cpp"
#endif

typedef class DPmixGGM *State;

class DPmixGGM {
  //-------- Variables ---------------
public:
  int n;
  int p;
  Real *X;
  myInt *xi;
  myInt L;
  LPGraph *graphlist;
  Real plp;  // log-prior of the partition
  Real *pll; // array of log-likelihood for the cluster
  Real alpha;

  //--------- Functions --------------
public:
  DPmixGGM(Real *data, myInt L_start, myInt n_obs, myInt p_model,
           Real edgeInclusionProb, ifstream &initfile);
  DPmixGGM(State a);
  ~DPmixGGM();

  void RandomStartAllXi(myInt L);
  void RandomStartAllG(myInt L, Real edgeInclusionProb);

  void RandomStart(myInt L, Real edgeInclusionProb);
  void InformedStart(ifstream &initfile);

  void ReadState(ifstream &initfile);
  void WriteState(ofstream &out, int itr);
  void CopyState(State a);

  Real partitionlogPrior(myInt theL, myInt *thexi, Real thealpha);
  Real lA(myInt the_n, myInt m, myInt k);

  Real cluster_k_loglikelihood(myInt k, myInt *thexi, LPGraph thegraph);
  Real predictiveDistribution(myInt k, myInt l, myInt *thexi, LPGraph thegraph);
};

//--------- This is the initializer ----------------------
DPmixGGM::DPmixGGM(Real *data, myInt L_start, myInt n_obs, myInt p_model,
                   Real edgeInclusionProb, ifstream &initfile) {
  myInt i;
  X = data;
  n = n_obs;
  p = p_model;
  L = L_start;
  xi = new myInt[n];
  graphlist = new LPGraph[L];
  for (i = 0; i < L; i++) {
    graphlist[i] = new Graph();
    graphlist[i]->InitGraph(p);
  };
  pll = new Real[L];

  // alpha = 20;
  alpha = 1;

#ifdef RANDOMSTART
  RandomStartAllXi(L);
  RandomStartAllG(L, edgeInclusionProb);
#else
  ReadState(initfile);
#endif

  plp = partitionlogPrior(L, xi, alpha);
  for (i = 0; i < L; i++) {
    pll[i] = cluster_k_loglikelihood(i, xi, graphlist[i]);
  }
}

DPmixGGM::DPmixGGM(State a) {
  myInt i;

  n = a->n;
  p = a->p;
  L = a->L;
  X = a->X;
  alpha = a->alpha;
  plp = a->plp;
  xi = new myInt[n];
  for (i = 0; i < n; i++) {
    xi[i] = a->xi[i];
  }
  graphlist = new LPGraph[L];
  pll = new Real[L];
  for (i = 0; i < L; i++) {
    graphlist[i] = new Graph();
    graphlist[i]->InitGraph(p);
    graphlist[i]->CopyGraph(a->graphlist[i]);
    pll[i] = a->pll[i];
  }
}

//-------- Destructor ------------------------------
DPmixGGM::~DPmixGGM() {
  myInt i;
  delete[] xi;
  for (i = 0; i < L; i++) {
    delete graphlist[i];
  };
  delete[] graphlist;
  delete[] pll;
}

void DPmixGGM::RandomStartAllXi(myInt L_start) {
  for (myInt i = 0; i < n; i++) {
    xi[i] = (myInt)(L_start * gsl_ran_flat(rnd, 0.0, 1.0)) / 1;
  }
}

void DPmixGGM::RandomStartAllG(myInt L_start, Real edgeInclusionProb) {
  myInt i, k, l;
  bool temp;

  for (i = 0; i < L_start; i++) {
    for (k = 0; k < p - 1; k++) {
      graphlist[i]->Edge[k][k] = 0;
      for (l = k + 1; l < p; l++) {
        temp = (gsl_ran_flat(rnd, 0.0, 1.0) < edgeInclusionProb);
        graphlist[i]->Edge[k][l] = temp;
        graphlist[i]->Edge[l][k] = temp;
      }
    }
    graphlist[i]->Edge[p - 1][p - 1] = 0;
    TurnFillInGraph(graphlist[i]);
    graphlist[i]->GenerateAllCliques();
  }
}

void DPmixGGM::RandomStart(myInt L_start, Real edgeInclusionProb) {
  myInt i;
  for (i = 0; i < L; i++) {
    delete graphlist[i];
  };
  delete graphlist;
  delete pll;
  L = L_start;
  graphlist = new LPGraph[L];
  for (i = 0; i < L; i++) {
    graphlist[i] = new Graph();
    graphlist[i]->InitGraph(p);
  };
  pll = new Real[L];

  RandomStartAllXi(L);
  RandomStartAllG(L, edgeInclusionProb);

  plp = partitionlogPrior(L, xi, alpha);
  for (i = 0; i < L; i++) {
    pll[i] = cluster_k_loglikelihood(i, xi, graphlist[i]);
  }
}

void DPmixGGM::InformedStart(ifstream &initfile) {
  ReadState(initfile);
  plp = partitionlogPrior(L, xi, alpha);
  for (myInt i = 0; i < L; i++) {
    pll[i] = cluster_k_loglikelihood(i, xi, graphlist[i]);
  }
}

void DPmixGGM::ReadState(ifstream &initfile) {
  myInt i, j, k, l;

  for (l = 0; l < L; l++) {
    delete graphlist[l];
  }
  delete[] graphlist;
  delete[] pll;

  int supern, superp;
  initfile >> supern;
  initfile >> superp;
  initfile >> L; // cout << "supern = " << supern << " superp = " << superp << "
                 // L = " << L << endl; fflush(stdout);
  for (i = 0; i < n; i++) {
    initfile >> xi[i];
    xi[i]--;
  };
  for (i = n; i < supern; i++) {
    initfile >> j;
  }
  graphlist = new LPGraph[L];
  pll = new Real[L];
  for (i = 0; i < L; i++) {
    graphlist[i] = new Graph();
    graphlist[i]->InitGraph(p);
    for (k = 0; k < p; k++) {
      for (l = 0; l < p; l++) {
        initfile >> graphlist[i]->Edge[k][l];
      };
      for (l = p; l < superp; l++) {
        initfile >> j;
      }
    }
    for (k = p; k < superp; k++) {
      for (l = 0; l < superp; l++) {
        initfile >> j;
      }
    }
    TurnFillInGraph(graphlist[i]);
    graphlist[i]->GenerateAllCliques();
  }

  // cout << "end of ReadState" << endl; fflush(stdout);
}

void DPmixGGM::WriteState(ofstream &out, int itr) {
  myInt i, j, l, q, r;
  Real score = plp;
  for (l = 0; l < L; l++) {
    score += pll[l];
  }

  out << L << " " << score << " " << itr << " ";
  for (i = 0; i < n; i++) {
    out << xi[i] << " ";
  }
  for (l = 0; l < L; l++) {
    for (q = 0; q < p - 1; q++) {
      for (r = q + 1; r < p; r++) {
        out << graphlist[l]->Edge[q][r] << " ";
      }
    }
  }
  for (l = 0; l < L; l++) {
    out << graphlist[l]->nCliques << " ";
    for (i = 0; i < (graphlist[l]->nCliques); i++) {
      out << graphlist[l]->CliquesDimens[i] << " ";
      for (j = 0; j < (graphlist[l]->CliquesDimens[i]); j++) {
        out << graphlist[l]->Cliques[i][j] << " ";
      }
    }

    out << graphlist[l]->nTreeEdges << " ";
    for (i = 0; i < (graphlist[l]->nTreeEdges); i++) {
      out << graphlist[l]->TreeEdgeA[i] << " " << graphlist[l]->TreeEdgeB[i]
          << " ";
    }

    out << (graphlist[l]->nSeparators) << " ";
    for (i = 0; i < (graphlist[l]->nSeparators); i++) {
      out << graphlist[l]->SeparatorsDimens[i] << " ";
      for (j = 0; j < (graphlist[l]->SeparatorsDimens[i]); j++) {
        out << graphlist[l]->Separators[i][j] << " ";
      }
    }
  }
  out << endl;
}

void DPmixGGM::CopyState(State a) {
  myInt i;
  myInt oldL = L;

  n = a->n;
  p = a->p;
  X = a->X;
  alpha = a->alpha;
  plp = a->plp;
  for (i = 0; i < n; i++) {
    xi[i] = a->xi[i];
  };
  if (L != a->L) {
    L = a->L;
    pll = new Real[L];
    for (i = 0; i < oldL; i++) {
      delete graphlist[i];
    };
    delete[] graphlist;
    graphlist = new LPGraph[L];
    for (i = 0; i < L; i++) {
      graphlist[i] = new Graph();
      graphlist[i]->InitGraph(p);
    }
  }
  for (i = 0; i < L; i++) {
    graphlist[i]->CopyGraph(a->graphlist[i]);
    pll[i] = a->pll[i];
  }
}

#ifdef JEFFREYS_PRIOR
struct f_params {
  int n;
  int m;
  int k;
};
double f(double beta, void *params) {
  struct f_params *iparams = (f_params *)params;
  int k = iparams->k;
  int n = iparams->n;
  int m = iparams->m;
  double s;
  double sum = 0;
  for (int j = 1; j < m; j++) {
    s = beta + j;
    sum += ((Real)j) / (s * s);
  }
  return exp(lgamma(beta) - lgamma(beta + n) + lgamma(n + 1.0) +
             (k - 0.5) * log(beta) + log(sqrt(sum)));
}

Real DPmixGGM::lA(myInt the_n, myInt m, myInt k) {
  struct f_params params = {the_n, m, k};
  F.function = &f;
  F.params = &params;
  double result, error;
  gsl_integration_qagiu(&F, 0.0, 1e-7, 1e-7, GSL_INTEGRATION_GRIDSIZE, w,
                        &result, &error);
  return log(result);
}
#endif

//--------------------------------------------------------------------
// This returns the partition prior probability -- normalising constant ignored
// assuming alpha is fixed maxL is maximum number of clusters, effective number
// of clusters can be smaller
Real DPmixGGM::partitionlogPrior(myInt maxL, myInt *thexi, Real thealpha) {
  myInt i, j;
  Real siz;
  Real pri = 0;
  myInt effectiveL = 0;
  for (i = 0; i < maxL; i++) {
    siz = 0.0;
    for (j = 0; j < n; j++) {
      if (thexi[j] == i)
        siz = siz + 1;
    };
    if (siz > 0) {
      pri += lgamma(siz);
      effectiveL++;
    }
  }

#ifdef JEFFREYS_PRIOR
  struct f_params params = {n, n, effectiveL};
  F.function = &f;
  F.params = &params;
  double result, error;
  gsl_integration_qagiu(&F, 0.0, 1e-7, 1e-7, GSL_INTEGRATION_GRIDSIZE, w,
                        &result, &error);
  return pri + log(result);
#else
  return pri + effectiveL * log(thealpha);
#endif
}

// This returns the likelihood of the observations in cluster k
Real DPmixGGM::cluster_k_loglikelihood(myInt k, myInt *thexi,
                                       LPGraph thegraph) {
  //-------- Variables, follows paper ---------------
  int i, j, ii;
  Real n0 = N0;
  Real *mu0 = new Real[p];
  for (i = 0; i < p; i++)
    mu0[i] = 0;
  Real *xbar = new Real[p];
  Real *mu_bar = new Real[p];
  Real *D_prior = new Real[p * (p + 1) / 2];
  Real *D_post = new Real[p * (p + 1) / 2];
  myInt n_sub = 0;
  Real J_G;
  Real Norm_terms;

  for (i = 0; i < n; i++) {
    if (thexi[i] == k)
      n_sub++;
  } // count the number in cluster k
  if (n_sub == 0) {
    delete[] D_post;
    delete[] D_prior;
    delete[] xbar;
    delete[] mu_bar;
    delete[] mu0;
    return (0);
  }
  // cout << "l = " << k << ", n_sub = " << n_sub << endl; fflush(stdout);

  //----------- Form the sufficient statistics ----------
  make_sub_means_and_cov(X, thexi, k, p, n, n_sub, xbar, D_post);

  //------------------ Update these ---------------------
  for (i = 0; i < p; i++)
    mu_bar[i] = (n_sub * xbar[i] + n0 * mu0[i]) / (n_sub + n0);
  for (i = 0; i < p * (p + 1) / 2; i++)
    D_prior[i] = 0;
  for (i = 0; i < p; i++)
    D_prior[i * (i + 1) / 2 + i] = 1;
  for (i = 0; i < p; i++)
    D_post[i * (i + 1) / 2 + i] += 1;

  //----------------- Factor in the mean for the matrix D_posterior
  //-------------
  for (i = 0; i < p; i++) {
    ii = i * (i + 1) / 2;
    for (j = 0; j <= i; j++) {
      D_post[ii + j] += (-(n_sub + n0) * mu_bar[i] * mu_bar[j] +
                         n_sub * xbar[i] * xbar[j] + n0 * mu0[i] * mu0[j]);
    }
  }

  //------------ Calculate the score
  //---------------------------------------------
  J_G = j_g_decomposable(thegraph, D_prior, D_post, DELTA0, n_sub, 0);
  Norm_terms =
      -(Real(n_sub * p) / 2) * log_2_pi + Real(p) / 2 * log(n0 / (n_sub + n0));

  delete[] D_post;
  delete[] D_prior;
  delete[] xbar;
  delete[] mu_bar;
  delete[] mu0;
  return (Norm_terms + J_G);
}

//--------------------------------------------------------------------
// This returns the predictive distribution score for observation k, if it were
// to belong to cluster l thegraph is the graph associated with cluster l
Real DPmixGGM::predictiveDistribution(myInt k, myInt l, myInt *thexi,
                                      LPGraph thegraph) {
  int i, j, ii;
  Real a;
  Real n0 = N0;
  Real *mu0 = new Real[p];
  Real *xbar = new Real[p];
  Real *mu_bar = new Real[p];
  Real *mu_tilde = new Real[p];
  for (i = 0; i < p; i++) {
    mu0[i] = 0;
  }
  Real *D_prior = new Real[p * (p + 1) / 2];
  Real *D_post = new Real[p * (p + 1) / 2];
  myInt n_sub = 0;
  Real coef;
  Real J_G;
  Real Norm_terms;

  for (i = 0; i < n; i++) {
    if (thexi[i] == l) {
      n_sub++;
    }
  } // count the number in this cluster
#ifdef JEFFREYS_PRIOR
  if (l == L) {
    coef = lA(n, n, L + 1) - lA(n - 1, n, L);
  } else if (n_sub > 0) {
    coef = log(n_sub) + lA(n, n, L) - lA(n - 1, n, L);
  } else {
    coef = 0;
  }
#else
  if (l == L) {
    coef = log(alpha);
  } else if (n_sub > 0) {
    coef = log(n_sub);
  } else {
    coef = 0;
  }
#endif

  //----------- Form the sufficient statistics ----------
  make_sub_means_and_cov(X, thexi, l, p, n, n_sub, xbar, D_prior);

  //------------------ Update these ---------------------
  for (i = 0; i < p; i++)
    mu_bar[i] = (n_sub * xbar[i] + n0 * mu0[i]) / (n_sub + n0);
  for (i = 0; i < p; i++)
    D_prior[i * (i + 1) / 2 + i] += 1;
  //----------------- Factor in the mean for the matrix D_prior -------------
  for (i = 0; i < p; i++) {
    ii = i * (i + 1) / 2;
    for (j = 0; j <= i; j++) {
      D_prior[ii + j] += -(n_sub + n0) * mu_bar[i] * mu_bar[j] +
                         n_sub * xbar[i] * xbar[j] + n0 * mu0[i] * mu0[j];
    }
  }

  //--------------  Get the posterior information ----------------------------
  for (i = 0; i < p * (p + 1) / 2; i++)
    D_post[i] = D_prior[i];
  for (i = 0; i < p; i++)
    mu_tilde[i] = (X[k * p + i] + (n_sub + n0) * mu_bar[i]) / (n_sub + n0 + 1);
  for (i = 0; i < p; i++) {
    ii = i * (i + 1) / 2;
    for (j = 0; j <= i; j++) {
      a = -(n_sub + 1 + n0) * mu_tilde[i] * mu_tilde[j] +
          X[k * p + i] * X[k * p + j] + (n_sub + n0) * mu_bar[i] * mu_bar[j];
      D_post[ii + j] += a;
    }
  }

  //------------ Calculate the score
  //---------------------------------------------
  J_G = j_g_decomposable(thegraph, D_prior, D_post, DELTA0 + n_sub, 1, 1);
  Norm_terms =
      -(p / 2) * log_2_pi + (p / 2) * log((n_sub + n0) / (n_sub + 1 + n0));

  delete[] D_post;
  delete[] D_prior;
  delete[] xbar;
  delete[] mu_bar;
  delete[] mu_tilde;
  delete[] mu0;
  return (coef + Norm_terms + J_G);
}