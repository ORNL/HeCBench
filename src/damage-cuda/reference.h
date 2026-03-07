#include <math.h>
#include <cstdlib>

void validate (
  const int block_size,
  const int m,
  const int n,
  const int *__restrict__ nlist,
  const int *__restrict__ family,
        int *__restrict__ n_neigh,
     double *__restrict__ damage)
{
  double *damage_ref = (double*) malloc (sizeof(double) * m);
  int *n_neigh_ref = (int*) malloc (sizeof(int) * m);

  for (int i = 0; i < m; i++) {
    int lb = i * block_size;
    int ub = lb + block_size;
    if (ub > n) ub = n;

    int sum = 0;
    for (int j = lb; j < ub; j++) {
      sum += nlist[j] != -1 ? 1 : 0;
    }

    int neighbours = sum;
    n_neigh_ref[i] = neighbours;
    damage_ref[i] = 1.0 - (double) neighbours / (double) (family[i]);
  }

  bool ok = true;
  for (int i = 0; i < m; i++) {
    if (n_neigh[i] != n_neigh_ref[i] ||
        fabs(damage[i] - damage_ref[i]) > 1e-6) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(damage_ref);
  free(n_neigh_ref);
}
