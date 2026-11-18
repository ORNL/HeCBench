#ifndef LID_DRIVEN_CAVITY_REFERENCE_H
#define LID_DRIVEN_CAVITY_REFERENCE_H

#include <cmath>
#include <cstdio>
#include <vector>

// Host check of the last time step of the solver.
//
// u0, v0, pr0, pb0 : velocities and red/black pressures at the start of the last step
// F, G, pr, pb, u, v: device results at the end of the last step
//
// u, v, F and G hold (num + 2) x (num + 2) values, and each pressure array
// holds (num + 2) x (num / 2 + 2) values.
//
// F and G are recomputed from u0 and v0. The pressure must satisfy the SOR
// tolerance for the right-hand side built from F and G, and its ghost cells
// must match the pressure boundary conditions. u and v, including
// their boundary values, are recomputed from F, G and the pressure.
template <typename T>
bool verify_last_step (const int num, const T dx, const T dy, const T dt,
                       const T Re_num, const T mix_param, const T gx, const T gy,
                       const T tol,
                       const T* u0, const T* v0, const T* pr0, const T* pb0,
                       const T* F, const T* G, const T* pr, const T* pb,
                       const T* u, const T* v)
{
  const int num_2 = num / 2;
  auto at  = [num](int i, int j) { return i * (num + 2) + j; };
  auto pat = [num_2](int i, int j) { return i * (num_2 + 2) + j; };

  // device code may contract multiply-adds and divide differently
  const T rtol = (sizeof(T) == sizeof(double)) ? (T)1e-10 : (T)1e-4;

  int errors = 0;
  auto check = [&](const char* name, int i, int j, T dev, T ref) {
    T err = std::fabs(dev - ref);
    if (!(err <= rtol * ((T)1 + std::fabs(ref)))) {
      if (errors++ < 10)
        printf("%s(%d, %d): %e (device) != %e (host)\n", name, i, j, (double)dev, (double)ref);
    }
  };

  // F and G
  for (int col = 1; col <= num; ++col) {
    for (int row = 1; row <= num; ++row) {
      if (col == num) {
        check("F", 0, row, F[at(0, row)], u0[at(0, row)]);
        check("F", num, row, F[at(num, row)], u0[at(num, row)]);
      } else {
        T u_ij = u0[at(col, row)];
        T u_ip1j = u0[at(col + 1, row)];
        T u_ijp1 = u0[at(col, row + 1)];
        T u_im1j = u0[at(col - 1, row)];
        T u_ijm1 = u0[at(col, row - 1)];
        T v_ij = v0[at(col, row)];
        T v_ip1j = v0[at(col + 1, row)];
        T v_ijm1 = v0[at(col, row - 1)];
        T v_ip1jm1 = v0[at(col + 1, row - 1)];

        T du2dx = (((u_ij + u_ip1j) * (u_ij + u_ip1j) - (u_im1j + u_ij) * (u_im1j + u_ij))
            + mix_param * (std::fabs(u_ij + u_ip1j) * (u_ij - u_ip1j)
              - std::fabs(u_im1j + u_ij) * (u_im1j - u_ij))) / (4 * dx);
        T duvdy = ((v_ij + v_ip1j) * (u_ij + u_ijp1) - (v_ijm1 + v_ip1jm1) * (u_ijm1 + u_ij)
            + mix_param * (std::fabs(v_ij + v_ip1j) * (u_ij - u_ijp1)
              - std::fabs(v_ijm1 + v_ip1jm1) * (u_ijm1 - u_ij))) / (4 * dy);
        T d2udx2 = (u_ip1j - (2 * u_ij) + u_im1j) / (dx * dx);
        T d2udy2 = (u_ijp1 - (2 * u_ij) + u_ijm1) / (dy * dy);

        T ref = u_ij + dt * (((d2udx2 + d2udy2) / Re_num) - du2dx - duvdy + gx);
        check("F", col, row, F[at(col, row)], ref);
      }

      if (row == num) {
        check("G", col, 0, G[at(col, 0)], v0[at(col, 0)]);
        check("G", col, num, G[at(col, num)], v0[at(col, num)]);
      } else {
        T u_ij = u0[at(col, row)];
        T u_ijp1 = u0[at(col, row + 1)];
        T u_im1j = u0[at(col - 1, row)];
        T u_im1jp1 = u0[at(col - 1, row + 1)];
        T v_ij = v0[at(col, row)];
        T v_ijp1 = v0[at(col, row + 1)];
        T v_ip1j = v0[at(col + 1, row)];
        T v_ijm1 = v0[at(col, row - 1)];
        T v_im1j = v0[at(col - 1, row)];

        T dv2dy = ((v_ij + v_ijp1) * (v_ij + v_ijp1) - (v_ijm1 + v_ij) * (v_ijm1 + v_ij)
            + mix_param * (std::fabs(v_ij + v_ijp1) * (v_ij - v_ijp1)
              - std::fabs(v_ijm1 + v_ij) * (v_ijm1 - v_ij))) / (4 * dy);
        T duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) - (u_im1j + u_im1jp1) * (v_im1j + v_ij)
            + mix_param * (std::fabs(u_ij + u_ijp1) * (v_ij - v_ip1j)
              - std::fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij))) / (4 * dx);
        T d2vdx2 = (v_ip1j - (2 * v_ij) + v_im1j) / (dx * dx);
        T d2vdy2 = (v_ijp1 - (2 * v_ij) + v_ijm1) / (dy * dy);

        T ref = v_ij + dt * (((d2vdx2 + d2vdy2) / Re_num) - dv2dy - duvdx + gy);
        check("G", col, row, G[at(col, row)], ref);
      }
    }
  }

  // Pressure residual. The ghost cells of the final pressure arrays are the
  // values the device used in its last residual evaluation.
  double p0_sum = 0.0, res_sum = 0.0;
  for (int col = 1; col <= num; ++col) {
    for (int row = 1; row <= num_2; ++row) {
      p0_sum += (double)pr0[pat(col, row)] * pr0[pat(col, row)]
              + (double)pb0[pat(col, row)] * pb0[pat(col, row)];

      // red point
      int k = (2 * row) - (col & 1);
      T p_ij = pr[pat(col, row)];
      T p_im1j = pb[pat(col - 1, row)];
      T p_ip1j = pb[pat(col + 1, row)];
      T p_ijm1 = pb[pat(col, row - (col & 1))];
      T p_ijp1 = pb[pat(col, row + ((col + 1) & 1))];
      T rhs = (((F[at(col, k)] - F[at(col - 1, k)]) / dx)
            +  ((G[at(col, k)] - G[at(col, k - 1)]) / dy)) / dt;
      double res = ((p_ip1j - (2 * p_ij) + p_im1j) / (dx * dx))
                 + ((p_ijp1 - (2 * p_ij) + p_ijm1) / (dy * dy)) - rhs;

      // black point
      k = (2 * row) - ((col + 1) & 1);
      p_ij = pb[pat(col, row)];
      p_im1j = pr[pat(col - 1, row)];
      p_ip1j = pr[pat(col + 1, row)];
      p_ijm1 = pr[pat(col, row - ((col + 1) & 1))];
      p_ijp1 = pr[pat(col, row + (col & 1))];
      rhs = (((F[at(col, k)] - F[at(col - 1, k)]) / dx)
          +  ((G[at(col, k)] - G[at(col, k - 1)]) / dy)) / dt;
      double res2 = ((p_ip1j - (2 * p_ij) + p_im1j) / (dx * dx))
                  + ((p_ijp1 - (2 * p_ij) + p_ijm1) / (dy * dy)) - rhs;

      res_sum += res * res + res2 * res2;
    }
  }
  double p0_norm = std::sqrt(p0_sum / ((double)num * num));
  if (p0_norm < 0.0001) p0_norm = 1.0;
  double norm_L2 = std::sqrt(res_sum / ((double)num * num)) / p0_norm;
  // the device sums the residual in a different order
  const double rslack = (sizeof(T) == sizeof(double)) ? 1e-6 : 1e-3;
  if (!(norm_L2 < tol * (1.0 + rslack))) {
    printf("Pressure residual %e exceeds the tolerance %e\n", norm_L2, (double)tol);
    errors++;
  }

  // Pressure boundary conditions. The ghost cells were set before the last
  // red-black sweep, so they lag their interior partners slightly.
  const double gtol = ((sizeof(T) == sizeof(double)) ? 1e-6 : 1e-3) * (1.0 + p0_norm);
  auto check_ghost = [&](const char* name, int i, int j, T ghost, T interior) {
    double err = std::fabs((double)ghost - interior);
    if (!(err <= gtol)) {
      if (errors++ < 10)
        printf("%s ghost (%d, %d): %e != %e\n", name, i, j, (double)ghost, (double)interior);
    }
  };
  for (int col = 1; col < num; col += 2) {
    check_ghost("pres_black", col, 0, pb[pat(col, 0)], pr[pat(col, 1)]);
    check_ghost("pres_red", col + 1, 0, pr[pat(col + 1, 0)], pb[pat(col + 1, 1)]);
    check_ghost("pres_red", col, num_2 + 1, pr[pat(col, num_2 + 1)], pb[pat(col, num_2)]);
    check_ghost("pres_black", col + 1, num_2 + 1, pb[pat(col + 1, num_2 + 1)], pr[pat(col + 1, num_2)]);
  }
  for (int row = 1; row <= num_2; ++row) {
    check_ghost("pres_black", 0, row, pb[pat(0, row)], pr[pat(1, row)]);
    check_ghost("pres_red", 0, row, pr[pat(0, row)], pb[pat(1, row)]);
    check_ghost("pres_black", num + 1, row, pb[pat(num + 1, row)], pr[pat(num, row)]);
    check_ghost("pres_red", num + 1, row, pr[pat(num + 1, row)], pb[pat(num, row)]);
  }

  // u and v
  std::vector<T> uh (u0, u0 + (num + 2) * (num + 2));
  std::vector<T> vh (v0, v0 + (num + 2) * (num + 2));
  for (int col = 1; col <= num; ++col) {
    for (int row = 1; row <= num_2; ++row) {
      int kr = (2 * row) - (col & 1);
      int kb = (2 * row) - ((col + 1) & 1);
      if (col != num) {
        uh[at(col, kr)] = F[at(col, kr)] - (dt * (pb[pat(col + 1, row)] - pr[pat(col, row)]) / dx);
        uh[at(col, kb)] = F[at(col, kb)] - (dt * (pr[pat(col + 1, row)] - pb[pat(col, row)]) / dx);
      }
      if (row != num_2) {
        vh[at(col, kr)] = G[at(col, kr)] - (dt * (pb[pat(col, row + ((col + 1) & 1))] - pr[pat(col, row)]) / dy);
        vh[at(col, kb)] = G[at(col, kb)] - (dt * (pr[pat(col, row + (col & 1))] - pb[pat(col, row)]) / dy);
      } else if ((col & 1) == 1) {
        vh[at(col, kr)] = G[at(col, kr)] - (dt * (pb[pat(col, row + ((col + 1) & 1))] - pr[pat(col, row)]) / dy);
      } else {
        vh[at(col, kb)] = G[at(col, kb)] - (dt * (pr[pat(col, row + (col & 1))] - pb[pat(col, row)]) / dy);
      }
    }
  }

  // velocity boundary conditions
  for (int ind = 1; ind <= num; ++ind) {
    uh[at(0, ind)] = 0;
    vh[at(0, ind)] = -vh[at(1, ind)];
    uh[at(num, ind)] = 0;
    vh[at(num + 1, ind)] = -vh[at(num, ind)];
    uh[at(ind, 0)] = -uh[at(ind, 1)];
    vh[at(ind, 0)] = 0;
    uh[at(ind, num + 1)] = 2 - uh[at(ind, num)];
    vh[at(ind, num)] = 0;

    if (ind == num) {
      uh[at(0, 0)] = 0;
      vh[at(0, 0)] = -vh[at(1, 0)];
      uh[at(0, num + 1)] = 0;
      vh[at(0, num + 1)] = -vh[at(1, num + 1)];
      uh[at(num, 0)] = 0;
      vh[at(num + 1, 0)] = -vh[at(num, 0)];
      uh[at(num, num + 1)] = 0;
      vh[at(num + 1, num + 1)] = -vh[at(num, num + 1)];
      uh[at(0, 0)] = -uh[at(0, 1)];
      vh[at(0, 0)] = 0;
      uh[at(num + 1, 0)] = -uh[at(num + 1, 1)];
      vh[at(num + 1, 0)] = 0;
      uh[at(0, num + 1)] = 2 - uh[at(0, num)];
      vh[at(0, num)] = 0;
      uh[at(num + 1, num + 1)] = 2 - uh[at(num + 1, num)];
      vh[at(ind, num + 1)] = 0;
    }
  }

  for (int i = 0; i <= num + 1; ++i) {
    for (int j = 0; j <= num + 1; ++j) {
      check("u", i, j, u[at(i, j)], uh[at(i, j)]);
      check("v", i, j, v[at(i, j)], vh[at(i, j)]);
    }
  }

  return errors == 0;
}

#endif
