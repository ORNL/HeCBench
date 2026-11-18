/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */
#include <unistd.h>
#include <thread>
#include <assert.h>
#include <chrono>
#include <cuda.h>

#include "kernel.h"
#include "support/partitioner.h"
#include "support/verify.h"

__constant__ float c_gaus[9] = {0.0625f, 0.125f, 0.0625f, 
                                0.1250f, 0.250f, 0.1250f, 
                                0.0625f, 0.125f, 0.0625f};
__constant__ int   c_sobx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int   c_soby[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

// https://github.com/smskelley/canny-opencl
// Gaussian Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
__global__ void 
gaussian_kernel(const unsigned char *__restrict__ data,
                unsigned char *__restrict__ out,
                const int rows, const int cols) 
{
  extern __shared__ int l_mem[];
  int* l_data = l_mem;

  const int L_SIZE = blockDim.x;
  int sum         = 0;
  const int l_row = threadIdx.y + 1;
  const int l_col = threadIdx.x + 1;
  const int g_row = blockIdx.y * blockDim.y + l_row;
  const int g_col = blockIdx.x * blockDim.x + l_col;

  const int pos = g_row * cols + g_col;

  // copy to local
  l_data[l_row * (L_SIZE + 2) + l_col] = data[pos];

  // top most row
  if(l_row == 1) {
      l_data[0 * (L_SIZE + 2) + l_col] = data[pos - cols];
      // top left
      if(l_col == 1)
          l_data[0 * (L_SIZE + 2) + 0] = data[pos - cols - 1];

      // top right
      else if(l_col == L_SIZE)
          l_data[0 * (L_SIZE + 2) + L_SIZE + 1] = data[pos - cols + 1];
  }
  // bottom most row
  else if(l_row == L_SIZE) {
      l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data[pos + cols];
      // bottom left
      if(l_col == 1)
          l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data[pos + cols - 1];

      // bottom right
      else if(l_col == L_SIZE)
          l_data[(L_SIZE + 1) * (L_SIZE + 2) + L_SIZE + 1] = data[pos + cols + 1];
  }

  if(l_col == 1)
      l_data[l_row * (L_SIZE + 2) + 0] = data[pos - 1];
  else if(l_col == L_SIZE)
      l_data[l_row * (L_SIZE + 2) + L_SIZE + 1] = data[pos + 1];

  __syncthreads();

  for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 3; j++) {
          sum += c_gaus[i*3+j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
      }
  }

  out[pos] = min(255, max(0, sum));
}

// Sobel kernel. Apply sobx and soby separately, then find the sqrt of their
//               squares.
// data:  image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out:   image output data (8B1C)
// theta: angle output data
__global__ void 
sobel_kernel(const unsigned char *__restrict__ data,
             unsigned char *__restrict__ out,
             unsigned char *__restrict__ theta,
             const int rows, const int cols)
{
  extern __shared__ int l_mem[];
  int* l_data = l_mem;

  // collect sums separately. we're storing them into floats because that
  // is what hypot and atan2 will expect.
  const int L_SIZE = blockDim.x;
  const float PI    = 3.14159265f;
  const int   l_row = threadIdx.y + 1;
  const int   l_col = threadIdx.x + 1;
  const int   g_row = blockIdx.y * blockDim.y + l_row;
  const int   g_col = blockIdx.x * blockDim.x + l_col;

  const int pos = g_row * cols + g_col;

  // copy to local
  l_data[l_row * (L_SIZE + 2) + l_col] = data[pos];

  // top most row
  if(l_row == 1) {
      l_data[0 * (L_SIZE + 2) + l_col] = data[pos - cols];
      // top left
      if(l_col == 1)
          l_data[0 * (L_SIZE + 2) + 0] = data[pos - cols - 1];

      // top right
      else if(l_col == L_SIZE)
          l_data[0 * (L_SIZE + 2) + (L_SIZE + 1)] = data[pos - cols + 1];
  }
  // bottom most row
  else if(l_row == L_SIZE) {
      l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data[pos + cols];
      // bottom left
      if(l_col == 1)
          l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data[pos + cols - 1];

      // bottom right
      else if(l_col == L_SIZE)
          l_data[(L_SIZE + 1) * (L_SIZE + 2) + (L_SIZE + 1)] = data[pos + cols + 1];
  }

  // left
  if(l_col == 1)
      l_data[l_row * (L_SIZE + 2) + 0] = data[pos - 1];
  // right
  else if(l_col == L_SIZE)
      l_data[l_row * (L_SIZE + 2) + (L_SIZE + 1)] = data[pos + 1];

  __syncthreads();

  float sumx = 0, sumy = 0, angle = 0;
  // find x and y derivatives
  for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 3; j++) {
          sumx += c_sobx[i*3+j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
          sumy += c_soby[i*3+j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
      }
  }

  // The output is now the square root of their squares, but they are
  // constrained to 0 <= value <= 255. Note that hypot is a built in function
  // defined as: hypot(x,y) = sqrt(x*x, y*y).
  out[pos] = min(255, max(0, (int)hypot(sumx, sumy)));

  // Compute the direction angle theta in radians
  // atan2 has a range of (-PI, PI) degrees
  angle = atan2(sumy, sumx);

  // If the angle is negative,
  // shift the range to (0, 2PI) by adding 2PI to the angle,
  // then perform modulo operation of 2PI
  if(angle < 0) {
      angle = fmod((angle + 2 * PI), (2 * PI));
  }

  // Round the angle to one of four possibilities: 0, 45, 90, 135 degrees
  // then store it in the theta buffer at the proper position
  //theta[pos] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
  if(angle <= PI / 8)
      theta[pos] = 0;
  else if(angle <= 3 * PI / 8)
      theta[pos] = 45;
  else if(angle <= 5 * PI / 8)
      theta[pos] = 90;
  else if(angle <= 7 * PI / 8)
      theta[pos] = 135;
  else if(angle <= 9 * PI / 8)
      theta[pos] = 0;
  else if(angle <= 11 * PI / 8)
      theta[pos] = 45;
  else if(angle <= 13 * PI / 8)
      theta[pos] = 90;
  else if(angle <= 15 * PI / 8)
      theta[pos] = 135;
  else
      theta[pos] = 0; // (angle <= 16*PI/8)
}

// Non-maximum Supression Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
// theta: angle input data
__global__ void 
non_max_supp_kernel(const unsigned char *__restrict__ data,
                          unsigned char *__restrict__ out, 
                    const unsigned char *__restrict__ theta,
                    const int rows, const int cols)
{
  extern __shared__ int l_mem[];
  int* l_data = l_mem;

  // These variables are offset by one to avoid seg. fault errors
  // As such, this kernel ignores the outside ring of pixels
  const int L_SIZE = blockDim.x;
  const int l_row = threadIdx.y + 1;
  const int l_col = threadIdx.x + 1;
  const int g_row = blockIdx.y * blockDim.y + l_row;
  const int g_col = blockIdx.x * blockDim.x + l_col;

  const int pos = g_row * cols + g_col;

  // copy to l_data
  l_data[l_row * (L_SIZE + 2) + l_col] = data[pos];

  // top most row
  if(l_row == 1) {
      l_data[0 * (L_SIZE + 2) + l_col] = data[pos - cols];
      // top left
      if(l_col == 1)
          l_data[0 * (L_SIZE + 2) + 0] = data[pos - cols - 1];

      // top right
      else if(l_col == L_SIZE)
          l_data[0 * (L_SIZE + 2) + (L_SIZE + 1)] = data[pos - cols + 1];
  }
  // bottom most row
  else if(l_row == L_SIZE) {
      l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = data[pos + cols];
      // bottom left
      if(l_col == 1)
          l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = data[pos + cols - 1];

      // bottom right
      else if(l_col == L_SIZE)
          l_data[(L_SIZE + 1) * (L_SIZE + 2) + (L_SIZE + 1)] = data[pos + cols + 1];
  }

  if(l_col == 1)
      l_data[l_row * (L_SIZE + 2) + 0] = data[pos - 1];
  else if(l_col == L_SIZE)
      l_data[l_row * (L_SIZE + 2) + (L_SIZE + 1)] = data[pos + 1];

  __syncthreads();

  unsigned char my_magnitude = l_data[l_row * (L_SIZE + 2) + l_col];

  // The following variables are used to address the matrices more easily
  switch(theta[pos]) {
    // A gradient angle of 0 degrees = an edge that is North/South
    // Check neighbors to the East and West
    case 0:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col + 1] || // east
            my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col - 1]) // west
        {
            out[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out[pos] = my_magnitude;
        }
        break;

    // A gradient angle of 45 degrees = an edge that is NW/SE
    // Check neighbors to the NE and SW
    case 45:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col + 1] || // north east
            my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col - 1]) // south west
        {
            out[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out[pos] = my_magnitude;
        }
        break;

    // A gradient angle of 90 degrees = an edge that is E/W
    // Check neighbors to the North and South
    case 90:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col] || // north
            my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col]) // south
        {
            out[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out[pos] = my_magnitude;
        }
        break;

    // A gradient angle of 135 degrees = an edge that is NE/SW
    // Check neighbors to the NW and SE
    case 135:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col - 1] || // north west
            my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col + 1]) // south east
        {
            out[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out[pos] = my_magnitude;
        }
        break;

    default: out[pos] = my_magnitude; break;
  }
}

// Hysteresis Threshold Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
__global__ void 
hyst_kernel(const unsigned char *__restrict__ data,
                  unsigned char *__restrict__ out,
            const int rows, const int cols)
{
  // Establish our high and low thresholds as floats
  float lowThresh  = 10;
  float highThresh = 70;

  // These variables are offset by one to avoid seg. fault errors
  // As such, this kernel ignores the outside ring of pixels
  const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int pos = row * cols + col;

  const unsigned char EDGE = 255;

  unsigned char magnitude = data[pos];

  if(magnitude >= highThresh)
      out[pos] = EDGE;
  else if(magnitude <= lowThresh)
      out[pos] = 0;
  else {
      float med = (highThresh + lowThresh) / 2;

      if(magnitude >= med)
          out[pos] = EDGE;
      else
          out[pos] = 0;
  }
}

// Params ---------------------------------------------------------------------
struct Params {

  int         device;
  int         n_gpu_threads;
  int         n_threads;
  int         n_warmup;
  int         n_reps;
  float       alpha;
  const char *file_name;
  const char *comparison_file;
  int         display = 0;

  Params(int argc, char **argv) {
    device          = 0;
    n_gpu_threads   = 16;
    n_threads       = 4;
    n_warmup        = 10;
    n_reps          = 100;
    alpha           = 0.2;
    file_name       = "../data/ced/input/peppa/";
    comparison_file = "../data/ced/output/peppa/";
    int opt;
    while((opt = getopt(argc, argv, "hd:i:t:w:r:a:f:c:x")) >= 0) {
      switch(opt) {
        case 'h':
          usage();
          exit(0);
          break;
        case 'i': n_gpu_threads   = atoi(optarg); break;
        case 't': n_threads       = atoi(optarg); break;
        case 'w': n_warmup        = atoi(optarg); break;
        case 'r': n_reps          = atoi(optarg); break;
        case 'a': alpha           = atof(optarg); break;
        case 'f': file_name       = optarg; break;
        case 'c': comparison_file = optarg; break;
        case 'x': display         = 1; break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
      }
    }
    if(alpha == 0.0) {
      assert(n_gpu_threads > 0 && "Invalid # of device threads!");
    } else if(alpha == 1.0) {
      assert(n_threads > 0 && "Invalid # of host threads!");
    } else if(alpha > 0.0 && alpha < 1.0) {
      assert(n_gpu_threads > 0 && "Invalid # of device threads!");
      assert(n_threads > 0 && "Invalid # of host threads!");
    } else {
      assert((n_gpu_threads > 0 || n_threads > 0) && "Invalid # of host + device workers!");
    }
#ifndef CHAI_OPENCV
    assert(display != 1 && "Compile with CHAI_OPENCV");
#endif
  }

  void usage() {
    fprintf(stderr,
        "\nUsage:  ./cedd [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -i <I>    # of device threads per block (default=16)"
        "\n    -t <T>    # of host threads (default=4)"
        "\n    -w <W>    # of untimed warmup iterations (default=10)"
        "\n    -r <R>    # of timed repetition iterations (default=100)"
        "\n"
        "\nData-partitioning-specific options:"
        "\n    -a <A>    fraction of input elements to process on host (default=0.2)"
        "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -f <F>    folder containing input video files (default=../data/ced/input/peppa/)"
        "\n    -c <C>    folder containing comparison files (default=../data/ced/output/peppa/)"
        "\n    -x        display output video (with CHAI_OPENCV)"
        "\n");
  }
};

// Input Data -----------------------------------------------------------------
void read_input(unsigned char** all_gray_frames, 
		int &rows, int &cols, int &in_size, const Params &p) {

  for(int task_id = 0; task_id < p.n_warmup + p.n_reps; task_id++) {

    char FileName[100];
    sprintf(FileName, "%s%d.txt", p.file_name, task_id);

    FILE *fp = fopen(FileName, "r");
    if(fp == NULL) {
      fprintf (stderr, "Failed to open the file %s. Exit\n.", FileName);
      exit(EXIT_FAILURE);
    }

    fscanf(fp, "%d\n", &rows);
    fscanf(fp, "%d\n", &cols);

    in_size = rows * cols * sizeof(unsigned char);
    all_gray_frames[task_id]    = (unsigned char *)malloc(in_size);
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[task_id][i * cols + j]);
      }
    }
    fclose(fp);
  }
}

int main(int argc, char **argv) {

  Params      p(argc, argv);

  // The maximum number of GPU threads is 1024 for certain GPUs
  const int max_gpu_threads = 256;
  assert(p.n_gpu_threads * p.n_gpu_threads <= max_gpu_threads && \
   "The thread block size is greater than the maximum thread block size");

  // read data from an 'input' directory which must be available
  const int n_frames = p.n_warmup + p.n_reps;
  unsigned char **all_gray_frames = 
    (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
  int     rows, cols, in_size;
  read_input(all_gray_frames, rows, cols, in_size, p);

  unsigned char **all_out_frames = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
  for(int i = 0; i < n_frames; i++) {
    all_out_frames[i] = (unsigned char *)malloc(in_size);
  }
  std::atomic_int *worklist = (std::atomic_int *)malloc(sizeof(std::atomic_int));
  if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
    worklist[0].store(0);
  }

  unsigned char* cpu_in_out = (unsigned char *)malloc(in_size);

  unsigned char *h_interm_cpu_proxy = (unsigned char *)malloc(in_size);
  unsigned char *h_theta_cpu_proxy  = (unsigned char *)malloc(in_size);

  unsigned char* d_in_out;
  cudaMalloc((void**)&d_in_out, sizeof(unsigned char)*in_size);
  
  unsigned char* d_interm_gpu_proxy;
  cudaMalloc((void**)&d_interm_gpu_proxy, sizeof(unsigned char)*in_size);

  unsigned char* d_theta_gpu_proxy;
  cudaMalloc((void**)&d_theta_gpu_proxy, sizeof(unsigned char)*in_size);

  CoarseGrainPartitioner partitioner = partitioner_create(n_frames, p.alpha, worklist);
  std::vector<std::thread> proxy_threads;

  auto t1 = std::chrono::high_resolution_clock::now();

  proxy_threads.push_back(std::thread([&]() {

      for(int task_id = gpu_first(&partitioner); gpu_more(&partitioner); task_id = gpu_next(&partitioner)) {

        // Copy next frame to device
        cudaMemcpy(d_in_out, all_gray_frames[task_id], in_size, cudaMemcpyHostToDevice);
     
        int threads = p.n_gpu_threads;
        dim3 grid ((cols-2)/threads, (rows-2)/threads);
        dim3 block (threads, threads);
        int smem_size = (threads+2)*(threads+2)*sizeof(int); 

        // call GAUSSIAN KERNEL
        gaussian_kernel<<<grid, block, smem_size>>>(d_in_out, d_interm_gpu_proxy, rows, cols);

        // call SOBEL KERNEL
        sobel_kernel<<<grid, block, smem_size>>>(d_interm_gpu_proxy, d_in_out, d_theta_gpu_proxy, rows, cols);

        // call NON-MAXIMUM SUPPRESSION KERNEL
        non_max_supp_kernel<<<grid, block, smem_size>>>(d_in_out, d_interm_gpu_proxy, d_theta_gpu_proxy, rows, cols);

        // call HYSTERESIS KERNEL
        hyst_kernel<<<grid, block, smem_size>>>(d_interm_gpu_proxy, d_in_out, rows, cols);

        // Copy from Device
        cudaMemcpy(all_out_frames[task_id], d_in_out, in_size, cudaMemcpyDeviceToHost);
      }

      for(int task_id = cpu_first(&partitioner); cpu_more(&partitioner); task_id = cpu_next(&partitioner)) {

        // Next frame
        memcpy(cpu_in_out, all_gray_frames[task_id], in_size);

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, cpu_in_out, h_interm_cpu_proxy, h_theta_cpu_proxy,
            rows, cols, p.n_threads, task_id);
        main_thread.join();

        memcpy(all_out_frames[task_id], cpu_in_out, in_size);

      }
  }));
  std::for_each(proxy_threads.begin(), proxy_threads.end(), [](std::thread &t) { t.join(); });

  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  printf("Total time %lf secs \n", total_time / 1.0e6);


#ifdef CHAI_OPENCV
  // Display the result
  if(p.display){
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
      cv::Mat out_frame = cv::Mat(rows, cols, CV_8UC1);
      memcpy(out_frame.data, all_out_frames[rep], in_size);
      if(!out_frame.empty())
        imshow("canny", out_frame);
      if(cv::waitKey(30) >= 0)
        break;
    }
  }
#endif

  // Verify answer
  int status = verify(all_out_frames, in_size, p.comparison_file, 
		  p.n_warmup + p.n_reps, rows, cols, rows, cols);

  // Release buffers
  cudaFree(d_in_out);
  cudaFree(d_interm_gpu_proxy);
  cudaFree(d_theta_gpu_proxy);

  free(cpu_in_out);
  free(h_interm_cpu_proxy);
  free(h_theta_cpu_proxy);
  for(int i = 0; i < n_frames; i++) {
    free(all_gray_frames[i]);
  }
  free(all_gray_frames);
  for(int i = 0; i < n_frames; i++) {
    free(all_out_frames[i]);
  }
  free(all_out_frames);
  free(worklist);

  if (status == 0) printf("PASS\n");
  return 0;
}
