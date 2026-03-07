/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include "vec_2d.hpp"
#include "utils.hpp"
#include <cstdlib>

template <typename T>
struct compare
{
  HOST_DEVICE
  inline bool operator()(const vec_2d<T>& a, const vec_2d<T>& b) const
  {
    return a.x * a.x + a.y * a.y < b.x * b.x + b.y * b.y;
  }
};

template <typename T>
void eval (const T bounding_box_size, const int repeat) {

  auto const [total_points, points] = generate_points(bounding_box_size);

  printf("Total number of points: %zu\n", total_points);

  thrust::device_vector<vec_2d<T>> d_points (points);

  vec_2d<T> min_point[2] = {{}, {}}, max_point[2] = {{}, {}};

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    auto const min_point_itr = thrust::min_element(
      thrust::device, d_points.begin(), d_points.end(), compare<T>());
    min_point[0] = points[thrust::distance(d_points.begin(), min_point_itr)];

    auto const max_point_itr = thrust::max_element(
      thrust::device, d_points.begin(), d_points.end(), compare<T>());
    max_point[0] = points[thrust::distance(d_points.begin(), max_point_itr)];
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of thrust:min() and thrust:max(): %f (us)\n",
         (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    auto const point_itr = thrust::minmax_element(
      thrust::device, d_points.begin(), d_points.end(), compare<T>());

    min_point[1] = points[thrust::distance(d_points.begin(), point_itr.first)];
    max_point[1] = points[thrust::distance(d_points.begin(), point_itr.second)];
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of thrust:min_max(): %f (us)\n",
         (time * 1e-3f) / repeat);

  // Verify
  auto const min_itr = std::min_element(
    points.begin(), points.end(), compare<T>());
  auto r_min_point = points[std::distance(points.begin(), min_itr)];

  auto const max_itr = std::max_element(
    points.begin(), points.end(), compare<T>());
  auto r_max_point = points[std::distance(points.begin(), max_itr)];

  bool ok = (min_point[0] == r_min_point) && (max_point[0] == r_max_point);
  ok &= (min_point[1] == r_min_point) && (max_point[1] == r_max_point);
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <bounding-box size> <repeat>\n", argv[0]);
    return 1;
  }
  const int size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  eval((float)size, repeat);

  return 0; 
}
