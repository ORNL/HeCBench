/*****************************************************************************
 Eigen-backed replacement for the subset of the GNU Scientific Library used by
 the XLQC benchmark, added so that the benchmark no longer links against the
 GPL-licensed GSL (see HeCBench issue #319).

 The names and signatures of the GSL entry points are kept so that the SCF code
 reads unchanged; the dense linear algebra is delegated to Eigen (MPL2):

   gsl_blas_dgemm       -> Eigen matrix product
   gsl_eigen_symmv      -> Eigen::SelfAdjointEigenSolver
   gsl_linalg_LU_decomp -> Eigen::PartialPivLU (partial pivoting, as in GSL)
   gsl_linalg_LU_solve  -> Eigen::PartialPivLU::solve

 Only what XLQC needs is implemented. Two deliberate deviations from GSL, both
 invisible to this benchmark: matrices are zero-filled on allocation rather
 than left uninitialised, and gsl_eigen_symmv leaves its input intact instead
 of destroying it.
 *****************************************************************************/

#ifndef GSL_COMPAT_H
#define GSL_COMPAT_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#define GSL_SUCCESS 0
#define GSL_EFAILED 5
#define GSL_EBADLEN 19

//===============================
// matrices and vectors
//===============================

typedef struct gsl_matrix_struct {
  size_t size1;
  size_t size2;
  Eigen::MatrixXd m;
  // Filled in by gsl_linalg_LU_decomp and consumed by gsl_linalg_LU_solve.
  // GSL keeps the factorisation in the matrix itself, so it lives here too.
  Eigen::PartialPivLU<Eigen::MatrixXd> *lu;
} gsl_matrix;

typedef struct gsl_vector_struct {
  size_t size;
  Eigen::VectorXd v;
} gsl_vector;

inline gsl_matrix *gsl_matrix_alloc(const size_t n1, const size_t n2) {
  gsl_matrix *a = new gsl_matrix;
  a->size1 = n1;
  a->size2 = n2;
  a->m = Eigen::MatrixXd::Zero(n1, n2);
  a->lu = NULL;
  return a;
}

inline gsl_matrix *gsl_matrix_calloc(const size_t n1, const size_t n2) {
  return gsl_matrix_alloc(n1, n2);
}

inline void gsl_matrix_free(gsl_matrix *a) {
  if (a == NULL) return;
  delete a->lu;
  delete a;
}

inline double gsl_matrix_get(const gsl_matrix *a, const size_t i,
                             const size_t j) {
  return a->m(i, j);
}

inline void gsl_matrix_set(gsl_matrix *a, const size_t i, const size_t j,
                           const double x) {
  a->m(i, j) = x;
}

inline void gsl_matrix_set_zero(gsl_matrix *a) { a->m.setZero(); }

inline int gsl_matrix_memcpy(gsl_matrix *dest, const gsl_matrix *src) {
  if (dest->size1 != src->size1 || dest->size2 != src->size2)
    return GSL_EBADLEN;
  dest->m = src->m;
  return GSL_SUCCESS;
}

inline gsl_vector *gsl_vector_alloc(const size_t n) {
  gsl_vector *x = new gsl_vector;
  x->size = n;
  x->v = Eigen::VectorXd::Zero(n);
  return x;
}

inline gsl_vector *gsl_vector_calloc(const size_t n) {
  return gsl_vector_alloc(n);
}

inline void gsl_vector_free(gsl_vector *x) { delete x; }

inline double gsl_vector_get(const gsl_vector *x, const size_t i) {
  return x->v(i);
}

inline void gsl_vector_set(gsl_vector *x, const size_t i, const double y) {
  x->v(i) = y;
}

inline void gsl_vector_set_zero(gsl_vector *x) { x->v.setZero(); }

//===============================
// BLAS level 3
//===============================

typedef enum {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
} CBLAS_TRANSPOSE_t;

// C = alpha * op(A) * op(B) + beta * C
inline int gsl_blas_dgemm(const CBLAS_TRANSPOSE_t TransA,
                          const CBLAS_TRANSPOSE_t TransB, const double alpha,
                          const gsl_matrix *A, const gsl_matrix *B,
                          const double beta, gsl_matrix *C) {
  const bool ta = (TransA != CblasNoTrans);
  const bool tb = (TransB != CblasNoTrans);

  const size_t ma = ta ? A->size2 : A->size1;
  const size_t na = ta ? A->size1 : A->size2;
  const size_t mb = tb ? B->size2 : B->size1;
  const size_t nb = tb ? B->size1 : B->size2;

  if (na != mb || ma != C->size1 || nb != C->size2) return GSL_EBADLEN;

  // Evaluated into a temporary so that C may alias A or B
  Eigen::MatrixXd prod(ma, nb);
  if (ta) {
    if (tb)
      prod.noalias() = A->m.transpose() * B->m.transpose();
    else
      prod.noalias() = A->m.transpose() * B->m;
  } else {
    if (tb)
      prod.noalias() = A->m * B->m.transpose();
    else
      prod.noalias() = A->m * B->m;
  }

  if (beta == 0.0)
    C->m = alpha * prod;
  else
    C->m = alpha * prod + beta * C->m;

  return GSL_SUCCESS;
}

//===============================
// symmetric eigenproblem
//===============================

typedef struct {
  size_t size;
} gsl_eigen_symmv_workspace;

inline gsl_eigen_symmv_workspace *gsl_eigen_symmv_alloc(const size_t n) {
  gsl_eigen_symmv_workspace *w = new gsl_eigen_symmv_workspace;
  w->size = n;
  return w;
}

inline void gsl_eigen_symmv_free(gsl_eigen_symmv_workspace *w) { delete w; }

// Eigenvalues in eval, corresponding eigenvectors in the columns of evec.
// Eigen already returns them in ascending order of eigenvalue.
inline int gsl_eigen_symmv(gsl_matrix *A, gsl_vector *eval, gsl_matrix *evec,
                           gsl_eigen_symmv_workspace *w) {
  (void)w;

  if (A->size1 != A->size2) return GSL_EBADLEN;
  if (eval->size != A->size1 || evec->size1 != A->size1 ||
      evec->size2 != A->size2)
    return GSL_EBADLEN;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A->m);
  if (es.info() != Eigen::Success) return GSL_EFAILED;

  eval->v = es.eigenvalues();
  evec->m = es.eigenvectors();

  return GSL_SUCCESS;
}

typedef enum {
  GSL_EIGEN_SORT_VAL_ASC,
  GSL_EIGEN_SORT_VAL_DESC,
  GSL_EIGEN_SORT_ABS_ASC,
  GSL_EIGEN_SORT_ABS_DESC
} gsl_eigen_sort_t;

inline int gsl_eigen_symmv_sort(gsl_vector *eval, gsl_matrix *evec,
                                const gsl_eigen_sort_t sort_type) {
  const size_t n = eval->size;
  if (evec->size2 != n) return GSL_EBADLEN;

  std::vector<size_t> idx(n);
  std::iota(idx.begin(), idx.end(), (size_t)0);

  const Eigen::VectorXd &e = eval->v;
  std::stable_sort(idx.begin(), idx.end(), [&](size_t i, size_t j) {
    switch (sort_type) {
      case GSL_EIGEN_SORT_VAL_ASC: return e(i) < e(j);
      case GSL_EIGEN_SORT_VAL_DESC: return e(i) > e(j);
      case GSL_EIGEN_SORT_ABS_ASC: return std::abs(e(i)) < std::abs(e(j));
      default: return std::abs(e(i)) > std::abs(e(j));
    }
  });

  Eigen::VectorXd eval_sorted(n);
  Eigen::MatrixXd evec_sorted(evec->size1, n);
  for (size_t k = 0; k < n; k++) {
    eval_sorted(k) = eval->v(idx[k]);
    evec_sorted.col(k) = evec->m.col(idx[k]);
  }

  eval->v = eval_sorted;
  evec->m = evec_sorted;

  return GSL_SUCCESS;
}

//===============================
// LU decomposition
//===============================

typedef struct {
  size_t size;
  std::vector<size_t> data;
} gsl_permutation;

inline gsl_permutation *gsl_permutation_alloc(const size_t n) {
  gsl_permutation *p = new gsl_permutation;
  p->size = n;
  p->data.resize(n);
  std::iota(p->data.begin(), p->data.end(), (size_t)0);
  return p;
}

inline gsl_permutation *gsl_permutation_calloc(const size_t n) {
  return gsl_permutation_alloc(n);
}

inline void gsl_permutation_free(gsl_permutation *p) { delete p; }

inline size_t gsl_permutation_get(const gsl_permutation *p, const size_t i) {
  return p->data[i];
}

// As in GSL, A is replaced by its LU factors and p by the row permutation.
inline int gsl_linalg_LU_decomp(gsl_matrix *A, gsl_permutation *p, int *signum) {
  *signum = 0;

  if (A->size1 != A->size2 || p->size != A->size1) return GSL_EBADLEN;

  delete A->lu;
  A->lu = new Eigen::PartialPivLU<Eigen::MatrixXd>(A->m);

  const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &P =
      A->lu->permutationP();

  for (size_t i = 0; i < p->size; i++)
    p->data[i] = (size_t)P.indices()(i);

  *signum = (int)P.determinant();

  A->m = A->lu->matrixLU();

  return GSL_SUCCESS;
}

inline int gsl_linalg_LU_solve(const gsl_matrix *LU, const gsl_permutation *p,
                               const gsl_vector *b, gsl_vector *x) {
  (void)p;

  if (LU->lu == NULL) {
    fprintf(stderr, "Error: gsl_linalg_LU_solve called before LU_decomp\n");
    return GSL_EFAILED;
  }
  if (b->size != LU->size1 || x->size != LU->size2) return GSL_EBADLEN;

  x->v = LU->lu->solve(b->v);

  return GSL_SUCCESS;
}

#endif // GSL_COMPAT_H
