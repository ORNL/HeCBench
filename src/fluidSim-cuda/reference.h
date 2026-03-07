double computefEq(double rho, double weight, const double dir[2], const double velocity[2])
{
  double u2 = velocity[0] * velocity[0] + velocity[1] * velocity[1];
  double eu = dir[0] * velocity[0] + dir[1] * velocity[1];
  return rho * weight * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
}

#ifdef VERIFY

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <algorithm>

// error bound
const double EPISON = 1e-3;

void reference (
  const int iterations,
  const double omega,
  const int *dims,
  const bool *h_type,
  double *rho,
  const double (*dir)[2],
  const double weight[9],
  const double *h_if0,
  const double *h_if1234,
  const double *h_if5678,
        double *h_of0,
        double *h_of1234,
        double *h_of5678)
{
  double *v_of0 = h_of0;
  double *v_of1234 = h_of1234;
  double *v_of5678 = h_of5678;

  const size_t temp = dims[0] * dims[1];
  const size_t dbl_size = temp * sizeof(double);
  const size_t dbl4_size = dbl_size * 4;

  double *v_if0 = (double*)malloc(dbl_size);
  double *v_if1234 = (double*)malloc(dbl4_size);
  double *v_if5678 = (double*)malloc(dbl4_size);

  // intermediate results
  double *v_ef0 = (double*)malloc(dbl_size);
  double *v_ef1234 = (double*)malloc(dbl4_size);
  double *v_ef5678 = (double*)malloc(dbl4_size);

  // initialize inputs and outputs
  memcpy(v_if0, h_if0, dbl_size);
  memcpy(v_if1234, h_if1234, dbl4_size);
  memcpy(v_if5678, h_if5678, dbl4_size);

  memcpy(v_of0, h_if0, dbl_size);
  memcpy(v_of1234, h_if1234, dbl4_size);
  memcpy(v_of5678, h_if5678, dbl4_size);

  for (int i = 0; i < iterations; i++) {
    // collide
    for (int y = 0; y < dims[1]; y++)
    {
      for (int x = 0; x < dims[0]; x++)
      {
        int pos = x + y * dims[0];

        if (h_type[pos] == 1) // Boundary
        {
          // Read input distributions
          v_ef0[pos] = v_if0[pos];
          double temp1 = v_if1234[pos * 4 + 2];
          double temp2 = v_if1234[pos * 4 + 3];
          double temp3 = v_if1234[pos * 4 + 0];
          double temp4 = v_if1234[pos * 4 + 1];
          v_ef1234[pos * 4 + 0] = temp1;
          v_ef1234[pos * 4 + 1] = temp2;
          v_ef1234[pos * 4 + 2] = temp3;
          v_ef1234[pos * 4 + 3] = temp4;

          temp1 = v_if5678[pos * 4 + 2];
          temp2 = v_if5678[pos * 4 + 3];
          temp3 = v_if5678[pos * 4 + 0];
          temp4 = v_if5678[pos * 4 + 1];
          v_ef5678[pos * 4 + 0] = temp1;
          v_ef5678[pos * 4 + 1] = temp2;
          v_ef5678[pos * 4 + 2] = temp3;
          v_ef5678[pos * 4 + 3] = temp4;

          rho[pos] = 0;
        }
        else // Fluid
        {
          double vel[2];

          // Calculate density from input distribution
          double den = v_if0[pos] + v_if1234[pos * 4 + 0] + 
                                    v_if1234[pos * 4 + 1] + 
                                    v_if1234[pos * 4 + 2] + 
                                    v_if1234[pos * 4 + 3] + 
                                    v_if5678[pos * 4 + 0] + 
                                    v_if5678[pos * 4 + 1] + 
                                    v_if5678[pos * 4 + 2] + 
                                    v_if5678[pos * 4 + 3];

          // Calculate velocity vector in x-direction
          vel[0] = v_if1234[pos * 4 + 0] * dir[1][0] + 
                   v_if1234[pos * 4 + 1] * dir[2][0] +
                   v_if1234[pos * 4 + 2] * dir[3][0] +
                   v_if1234[pos * 4 + 3] * dir[4][0] +
                   v_if5678[pos * 4 + 0] * dir[5][0] +
                   v_if5678[pos * 4 + 1] * dir[6][0] +
                   v_if5678[pos * 4 + 2] * dir[7][0] +
                   v_if5678[pos * 4 + 3] * dir[8][0];

          // Calculate velocity vector in y-direction
          vel[1] = v_if1234[pos * 4 + 0] * dir[1][1] +
                   v_if1234[pos * 4 + 1] * dir[2][1] +
                   v_if1234[pos * 4 + 2] * dir[3][1] +
                   v_if1234[pos * 4 + 3] * dir[4][1] +
                   v_if5678[pos * 4 + 0] * dir[5][1] +
                   v_if5678[pos * 4 + 1] * dir[6][1] +
                   v_if5678[pos * 4 + 2] * dir[7][1] +
                   v_if5678[pos * 4 + 3] * dir[8][1];

          vel[0] /= den;
          vel[1] /= den;

          // Calculate Equivalent distribution
          v_ef0[pos]            = computefEq(den, weight[0], dir[0], vel);
          v_ef1234[pos * 4 + 0] = computefEq(den, weight[1], dir[1], vel);
          v_ef1234[pos * 4 + 1] = computefEq(den, weight[2], dir[2], vel);
          v_ef1234[pos * 4 + 2] = computefEq(den, weight[3], dir[3], vel);
          v_ef1234[pos * 4 + 3] = computefEq(den, weight[4], dir[4], vel);
          v_ef5678[pos * 4 + 0] = computefEq(den, weight[5], dir[5], vel);
          v_ef5678[pos * 4 + 1] = computefEq(den, weight[6], dir[6], vel);
          v_ef5678[pos * 4 + 2] = computefEq(den, weight[7], dir[7], vel);
          v_ef5678[pos * 4 + 3] = computefEq(den, weight[8], dir[8], vel);

          v_ef0[pos] = (1 - omega) * v_if0[pos] + omega * v_ef0[pos];
          v_ef1234[pos * 4 + 0] = (1 - omega) * v_if1234[pos * 4 + 0] + omega * v_ef1234[pos * 4 + 0];
          v_ef1234[pos * 4 + 1] = (1 - omega) * v_if1234[pos * 4 + 1] + omega * v_ef1234[pos * 4 + 1];
          v_ef1234[pos * 4 + 2] = (1 - omega) * v_if1234[pos * 4 + 2] + omega * v_ef1234[pos * 4 + 2];
          v_ef1234[pos * 4 + 3] = (1 - omega) * v_if1234[pos * 4 + 3] + omega * v_ef1234[pos * 4 + 3];
          v_ef5678[pos * 4 + 0] = (1 - omega) * v_if5678[pos * 4 + 0] + omega * v_ef5678[pos * 4 + 0];
          v_ef5678[pos * 4 + 1] = (1 - omega) * v_if5678[pos * 4 + 1] + omega * v_ef5678[pos * 4 + 1];
          v_ef5678[pos * 4 + 2] = (1 - omega) * v_if5678[pos * 4 + 2] + omega * v_ef5678[pos * 4 + 2];
          v_ef5678[pos * 4 + 3] = (1 - omega) * v_if5678[pos * 4 + 3] + omega * v_ef5678[pos * 4 + 3];
        }
      }
    }

    // Propagate
    for (int y = 1; y < dims[1]-1; y++)
    {
      for (int x = 1; x < dims[0]-1; x++)
      {
        int src_pos = x + dims[0] * y;
        for (int k=0; k<9; k++)
        {
          // New positions to write
          int nx = x + (int)dir[k][0];
          int ny = y + (int)dir[k][1];
          int dst_pos = nx + dims[0] * ny;
          switch(k)
          {
            case 0:
              v_of0[dst_pos] = v_ef0[src_pos];
              break;
            case 1:
              v_of1234[dst_pos * 4 + 0] = v_ef1234[src_pos * 4 + 0];
              break;
            case 2:
              v_of1234[dst_pos * 4 + 1] = v_ef1234[src_pos * 4 + 1];
              break;
            case 3:
              v_of1234[dst_pos * 4 + 2] = v_ef1234[src_pos * 4 + 2];
              break;
            case 4:
              v_of1234[dst_pos * 4 + 3] = v_ef1234[src_pos * 4 + 3];
              break;
            case 5:
              v_of5678[dst_pos * 4 + 0] = v_ef5678[src_pos * 4 + 0];
              break;
            case 6:
              v_of5678[dst_pos * 4 + 1] = v_ef5678[src_pos * 4 + 1];
              break;
            case 7:
              v_of5678[dst_pos * 4 + 2] = v_ef5678[src_pos * 4 + 2];
              break;
            case 8:
              v_of5678[dst_pos * 4 + 3] = v_ef5678[src_pos * 4 + 3];
              break;
          }
        }
      }
    }

    std::swap(v_if0, v_of0);
    std::swap(v_if1234, v_of1234);
    std::swap(v_if5678, v_of5678);
  }

  // Copy results to the output arrays
  memcpy(h_of0, v_if0, dbl_size);
  memcpy(h_of1234, v_if1234, dbl4_size);
  memcpy(h_of5678, v_if5678, dbl4_size);

  // Release input and intermediate arrays
  if (iterations % 2) {
    std::swap(v_if0, v_of0);
    std::swap(v_if1234, v_of1234);
    std::swap(v_if5678, v_of5678);
  }

  if(v_if0)    free(v_if0);
  if(v_if1234) free(v_if1234);
  if(v_if5678) free(v_if5678);
  if(v_ef0)    free(v_ef0);
  if(v_ef1234) free(v_ef1234);
  if(v_ef5678) free(v_ef5678);
}

void verify(
  const int *dims,
  const double *h_of0,
  const double *h_of1234,
  const double *h_of5678,
  const double *v_of0,
  const double *v_of1234,
  const double *v_of5678)
{
  int flag0 = 0;
  for (int y = 0; y < dims[1]; y++)
  {
    for (int x = 0; x < dims[0]; x++)
    {
      int pos = x + y * dims[0];
      if(h_of0[pos] - v_of0[pos] > EPISON)
      {
        //std::cout << "of0 @" << pos << " device:" << h_of0[pos] 
        //          << " host:" << v_of0[pos] << std::endl;
        flag0 = 1;
        break;
      }
    }
  }

  int flag1234 = 0;
  for (int y = 0; y < dims[1]; y++)
  {
    for (int x = 0; x < dims[0]; x++)
    {
      int pos = x + y * dims[0];
      if(h_of1234[pos] - v_of1234[pos] > EPISON)
      {
        flag1234 = 1;
        break;
      } 
    }
  }

  int flag5678 = 0;
  for (int y = 0; y < dims[1]; y++)
  {
    for (int x = 0; x < dims[0]; x++)
    {
      int pos = x + y * dims[0];
      if(h_of5678[pos] - v_of5678[pos] > EPISON)
      {
        flag5678 = 1;
        break;
      } 
    }
  }

  // Verify flag
  bool ok = true;
  if(flag0 || flag1234 || flag5678) ok = false;
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
}

#endif
