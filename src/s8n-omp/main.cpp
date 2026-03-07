#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "reference.h"

void k_cube_select(int b, int n, int radius, const int* in, int* out) {
  #pragma omp target teams distribute num_teams(b) 
  for (int batch_idx = 0; batch_idx < b; batch_idx++) {
    auto xyz = in + batch_idx * n * 3;
    auto idx_out = out + batch_idx * n * 8;
    #pragma omp parallel for num_threads(512)
    for(int i = 0; i < n; i++) {
      int temp_dist[8];
      int x = xyz[i * 3];
      int y = xyz[i * 3 + 1];
      int z = xyz[i * 3 + 2];
      for(int j = 0; j < 8;j ++) {
        temp_dist[j] = radius;
        idx_out[i * 8 + j] = i; // if not found, just return itself..
      }
      for(int j = 0; j < n; j ++) {
        if(i != j) continue;
        int tx = xyz[j * 3];
        int ty = xyz[j * 3 + 1];
        int tz = xyz[j * 3 + 2];
        int dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
        if(dist > radius) continue;
        int _x = (tx > x);
        int _y = (ty > y);
        int _z = (tz > z);
        int temp_idx = _x * 4 + _y * 2 + _z;
        if(dist < temp_dist[temp_idx]) {
          idx_out[i * 8 + temp_idx] = j;
          temp_dist[temp_idx] = dist;
        }
      }
    }
  }
}

void k_cube_select_two(int b, int n, int radius, const int* in, int* out) {
  #pragma omp target teams distribute num_teams(b) 
  for (int batch_idx = 0; batch_idx < b; batch_idx++) {
    auto xyz = in + batch_idx * n * 3;
    auto idx_out = out + batch_idx * n * 16;
    #pragma omp parallel for num_threads(512)
    for(int i = 0; i < n; i++) {
      int temp_dist[16];
      int x = xyz[i * 3];
      int y = xyz[i * 3 + 1];
      int z = xyz[i * 3 + 2];
      for(int j = 0; j < 16;j ++) {
        temp_dist[j] = radius;
        idx_out[i * 16 + j] = i; // if not found, just return itself..
      }
      for(int j = 0; j < n; j ++) {
        if(i == j) continue;
        int tx = xyz[j * 3];
        int ty = xyz[j * 3 + 1];
        int tz = xyz[j * 3 + 2];
        int dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
        if(dist > radius) continue;
        int _x = (tx > x);
        int _y = (ty > y);
        int _z = (tz > z);
        int temp_idx = _x * 8 + _y * 4 + _z * 2;
        bool flag = false;
        for(int k = 0; k < 2; k ++) {
          if (dist < temp_dist[temp_idx + k]) {
            flag = true;
          }
          if (flag) {
            for (int kk = 1; kk >= k + 1; kk --) {
              idx_out[i * 16 + temp_idx + kk] = idx_out[i * 16 + temp_idx + kk - 1];
              temp_dist[temp_idx + kk] = temp_dist[temp_idx + kk - 1];
            }
            idx_out[i * 16 + temp_idx + k] = j;
            temp_dist[temp_idx + k] = dist;
            break;
          }
        }
      }
    }
  }
}

void k_cube_select_four(int b, int n, int radius, const int* in, int* out) {
  #pragma omp target teams distribute num_teams(b) 
  for (int batch_idx = 0; batch_idx < b; batch_idx++) {
    auto xyz = in + batch_idx * n * 3;
    auto idx_out = out + batch_idx * n * 32;
    #pragma omp parallel for num_threads(512)
    for(int i = 0; i < n; i++) {
      int temp_dist[32];
      int x = xyz[i * 3];
      int y = xyz[i * 3 + 1];
      int z = xyz[i * 3 + 2];
      for(int j = 0; j < 32;j ++) {
        temp_dist[j] = radius;
        idx_out[i * 32 + j] = i; // if not found, just return itself..
      }
      for(int j = 0; j < n; j ++) {
        if(i == j) continue;
        int tx = xyz[j * 3];
        int ty = xyz[j * 3 + 1];
        int tz = xyz[j * 3 + 2];
        int dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
        if(dist > radius) continue;
        int _x = (tx > x);
        int _y = (ty > y);
        int _z = (tz > z);
        int temp_idx = _x * 16 + _y * 8 + _z * 4;
        bool flag = false;
        for(int k = 0; k < 4; k ++) {
          if (dist < temp_dist[temp_idx + k]) {
            flag = true;
          }
          if (flag) {
            for (int kk = 3; kk >= k + 1; kk --) {
              idx_out[i * 32 + temp_idx + kk] = idx_out[i * 32 + temp_idx + kk - 1];
              temp_dist[temp_idx + kk] = temp_dist[temp_idx + kk - 1];
            }
            idx_out[i * 32 + temp_idx + k] = j;
            temp_dist[temp_idx + k] = dist;
            break;
          }
        }
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of batches> <number of points> <repeat>\n", argv[0]);
    return 1;
  }
  const int b = atoi(argv[1]);
  const int n = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  int input_size = b * n * 3;
  size_t input_size_bytes = input_size * sizeof(int);

  size_t output_size = b * n * 8;
  size_t output_size_bytes = output_size * sizeof(int);

  const int radius = 512;

  int *h_xyz;
  int *h_out, *h_out2, *h_out4;
  int *r_out, *r_out2, *r_out4;

  h_xyz = (int*) malloc (input_size_bytes);
  h_out = (int*) malloc (output_size_bytes);
  r_out = (int*) malloc (output_size_bytes);
  h_out2 = (int*) malloc (2 * output_size_bytes);
  r_out2 = (int*) malloc (2 * output_size_bytes);
  h_out4 = (int*) malloc (4 * output_size_bytes);
  r_out4 = (int*) malloc (4 * output_size_bytes);

  std::default_random_engine g (123);
  std::uniform_int_distribution<> distr (-256, 255);
  for (int i = 0; i < input_size; i++) {
    h_xyz[i] = distr(g);
  }

  #pragma omp target data map (to: h_xyz[0:input_size]) \
                          map (alloc: h_out[0:output_size], \
                                      h_out2[0:output_size*2], \
                                      h_out4[0:output_size*4])
  {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
     k_cube_select(b, n, radius, h_xyz, h_out); 
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of select kernel: %f (us)\n", (time * 1e-3f) / repeat);

    #pragma omp target update from (h_out[0:output_size])
    cube_select(b, n, radius, h_xyz, r_out);
    int error = memcmp(h_out, r_out, output_size_bytes);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      k_cube_select_two(b, n, radius, h_xyz, h_out2); 
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of select2 kernel: %f (us)\n", (time * 1e-3f) / repeat);

    #pragma omp target update from (h_out2[0:output_size * 2])
    cube_select_two(b, n, radius, h_xyz, r_out2);
    error += memcmp(h_out2, r_out2, 2 * output_size_bytes);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      k_cube_select_four(b, n, radius, h_xyz, h_out4); 
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of select4 kernel: %f (us)\n", (time * 1e-3f) / repeat);

    #pragma omp target update from (h_out4[0:output_size * 4])
    cube_select_four(b, n, radius, h_xyz, r_out4);
    error += memcmp(h_out4, r_out4, 4 * output_size_bytes);

    printf("%s\n", error ? "FAIL" : "PASS");
    if (error) exit(1);
  }
  
  free(h_xyz);
  free(h_out);
  free(h_out2);
  free(h_out4);
  free(r_out);
  free(r_out2);
  free(r_out4);
  return 0;
}
