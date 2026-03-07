#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "utils.hpp"

template <typename T>
void coordinates_transform(const int num_coords, const int repeat)
{
  thrust::host_vector<lonlat_2d<T>> h_input (num_coords);
  thrust::host_vector<cartesian_2d<T>> h_output (num_coords);
  thrust::host_vector<cartesian_2d<T>> h_ref_output (num_coords);

  lonlat_2d<T> h_origin;
  h_origin.x = (T)90; // arbitrary valid values
  h_origin.y = (T)45;

  // valid longitude [-180, 180] and latitude [-90, 90]
  for (int i = 0; i < num_coords; i++) {
    h_input[i].x = (T)(rand() % 360 - 180);
    h_input[i].y = (T)(rand() % 180 - 90);
  }

  thrust::device_vector<lonlat_2d<T>> d_input = h_input;
  thrust::device_vector<cartesian_2d<T>> d_output (num_coords);
 
  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    thrust::transform(d_input.cbegin(), d_input.cend(),
                      d_output.begin(), to_cartesian_functor<T>(h_origin));
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of device transform: %f (us)\n", (time * 1e-3f) / repeat);

  h_output = d_output;  // copy results from device to host

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < 10; i++) {
    std::transform(h_input.cbegin(), h_input.cend(),
                   h_ref_output.begin(), to_cartesian_functor<T>(h_origin));
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of host transform: %f (us)\n", (time * 1e-3f) / 10);

  bool ok = true;
  for (int i = 0; i < num_coords; i++) {
    if (!(h_output[i] == h_ref_output[i])) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of coordinates> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_coords = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  srand(123);

  printf("\nDouble-precision coordinates transform\n");
  coordinates_transform<double>(num_coords, repeat);

  printf("\nSingle-precision coordinates transform\n");
  coordinates_transform<float>(num_coords, repeat);

  return 0;
}
