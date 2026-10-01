#include <cmath>
// gwish.cpp:  This is a collection of device functions that manipulate graph
// objects according to the G-Wishart distribution. Note: this depends on the
// graph.cpp library Chiranjit Mukherjee : chiranjit@soe.ucsc.edu -- based on
// Alex Lenkoski : lenkoski@stat.washington.edu

#define GWISH_CPP
#ifndef GRAPH_H
#include "graph.h"
#endif

void log_det(int p, Real *A, Real *result) {
  int i, j, k;
  Real temp;
  *result = 0;
  Real br;

  for (i = 0; i < p; i++) {
    for (j = i; j < p; j++) {
      temp = A[j * (j + 1) / 2 + i];
      for (k = i; k > 0; k--) {
        temp = temp - A[j * (j + 1) / 2 + k - 1] * A[i * (i + 1) / 2 + k - 1];
      };
      if (i == j) {
        if (temp <= 0.0) {
          *result = NEG_INF;
          return;
        } else {
          br = sqrt(temp);
        }
      }
      A[j * (j + 1) / 2 + i] = temp / br;
    }
  }

#ifdef ISFLOAT
  for (i = 0; i < p; i++) {
    *result += logf(A[i * (i + 1) / 2 + i]);
  };
  *result = 2 * (*result);
#else
  for (i = 0; i < p; i++) {
    *result += log(A[i * (i + 1) / 2 + i]);
  };
  *result = 2 * (*result);
#endif
}

// Computes the normalizing constant of a G-Wishart distribution for a full
// p-dimensional graph with parameters delta and D : br p (reusable)
Real gwish_nc_complete(myInt delta, int p, Real *D, bool flag) {
  Real d, c, a, g;
  Real dblDelta; // Recasting the inputs makes life easier below
  Real dblP;
  myInt i;
  c = 0.0;
  a = 0.0;
  g = 0.0;
  d = 0.0;
  dblDelta = delta;
  dblP = p;

  if (flag) {
    log_det(p, D, &d);
  }
  a = (dblDelta + dblP - 1) / 2.0;
  d = a * d;
  c = dblP * a * log_2;
  g = dblP * (dblP - 1) * log_pi_over_4;

  int signp;
#ifdef ISFLOAT
  for (i = 0; i < p; i++)
    g += lgammaf_r(a - (Real)i / 2.0, &signp);
#else
  for (i = 0; i < p; i++)
    g += lgamma_r(a - (Real)i / 2.0, &signp);
#endif

  return (-d + c + g);
}

// Utility function for making submatrices
void make_sub_mat_dbl(int p, int p_sub, myInt *sub, Real *A, Real *B) {
  int i, j;
  for (i = 0; i < p_sub; i++) {
    for (j = 0; j <= i; j++) {
      B[i * (i + 1) / 2 + j] =
          ((sub[i] >= sub[j]) ? A[sub[i] * (sub[i] + 1) / 2 + sub[j]]
                              : A[sub[j] * (sub[j] + 1) / 2 + sub[i]]);
    }
  }
}

// Utility function for making a mean vector and a covariance matrix over a
// subset of the dataset indicated by the vector sub.
void make_sub_means_and_cov(Real *X, myInt *sub, myInt sub_match, int p,
                            myInt n, myInt n_sub, Real *means, Real *D) {
  int i, j, k, ii;
  for (i = 0; i < p; i++) {
    means[i] = 0.0;
  };
  for (i = 0; i < p * (p + 1) / 2; i++) {
    D[i] = 0.0;
  }

  if (n_sub == 0) {
    return;
  }
  for (i = 0; i < p; i++) {
    for (k = 0; k < n; k++) {
      if (sub[k] == sub_match) {
        means[i] += (X[k * p + i] / (Real)n_sub);
      }
    }
  }
  if (n_sub < 2) {
    return;
  }

  for (i = 0; i < p; i++) {
    ii = i * (i + 1) / 2;
    for (j = 0; j <= i; j++) {
      for (k = 0; k < n; k++) {
        if (sub[k] == sub_match) {
          D[ii + j] += (X[k * p + i] - means[i]) * (X[k * p + j] - means[j]);
        }
      }
    }
  }

  return;
}

Real j_g_decomposable(LPGraph graph, Real *D_prior, Real *D_post, myInt delta,
                      myInt n, bool flag) {
  Real mypost = 0;
  int p = graph->nVertices;
  myInt i;
  myInt sub_p;
  Real *sub_D = new Real[2 * p * p];

  //----- First loop through all the prime components (cliques since we're
  //decomposable) ---
  for (i = 0; i < graph->nCliques; i++) {
    sub_p = graph->CliquesDimens[i];
    if (flag) {
      make_sub_mat_dbl(p, sub_p, graph->Cliques[i], D_prior, sub_D);
    };
    mypost -= gwish_nc_complete(delta, sub_p, sub_D, flag);
    make_sub_mat_dbl(p, sub_p, graph->Cliques[i], D_post, sub_D);
    mypost += gwish_nc_complete(delta + n, sub_p, sub_D, 1);
  }

  //------- Now subtract off the separators -----------------------------------
  for (i = 0; i < graph->nSeparators; i++) {
    sub_p = graph->SeparatorsDimens[i];
    if (flag) {
      make_sub_mat_dbl(p, sub_p, graph->Separators[i], D_prior, sub_D);
    };
    mypost += gwish_nc_complete(delta, sub_p, sub_D, flag);
    make_sub_mat_dbl(p, sub_p, graph->Separators[i], D_post, sub_D);
    mypost -= gwish_nc_complete(delta + n, sub_p, sub_D, 1);
  }

  delete[] sub_D;
  return (mypost);
}