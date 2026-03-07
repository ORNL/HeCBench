// Designed by: Amir Yazdanbakhsh
// Date: March 26th - 2015
// Alternative Computing Technologies Lab.
// Georgia Institute of Technology

#include <fstream>
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <sycl/sycl.hpp>

#define MAX_LOOP 25
#define MAX_DIFF 0.15f
#define NUM_JOINTS 3
#define PI 3.14159265358979f
#define NUM_JOINTS_P1 (NUM_JOINTS + 1)
#define BLOCK_SIZE 128

void invkin_cpu(float *xTarget_in, float *yTarget_in, float *angles, int size)
{
  for (int idx = 0; idx < size; idx++)
  {
    float angle_out[NUM_JOINTS];
    float xData[NUM_JOINTS_P1];
    float yData[NUM_JOINTS_P1];

    float curr_xTargetIn = xTarget_in[idx];
    float curr_yTargetIn = yTarget_in[idx];

    for(int i = 0; i < NUM_JOINTS; i++)
    {
      angle_out[i] = 0.0;
    }
    float angle;
    for (int i = 0 ; i < NUM_JOINTS_P1; i++)
    {
      xData[i] = i;
      yData[i] = 0.f;
    }

    for(int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++)
    {
      for (int iter = NUM_JOINTS; iter > 0; iter--) 
      {
        float pe_x = xData[NUM_JOINTS];
        float pe_y = yData[NUM_JOINTS];
        float pc_x = xData[iter-1];
        float pc_y = yData[iter-1];
        float diff_pe_pc_x = pe_x - pc_x;
        float diff_pe_pc_y = pe_y - pc_y;
        float diff_tgt_pc_x = curr_xTargetIn - pc_x;
        float diff_tgt_pc_y = curr_yTargetIn - pc_y;
        float len_diff_pe_pc = sqrtf(diff_pe_pc_x * diff_pe_pc_x + diff_pe_pc_y * diff_pe_pc_y);
        float len_diff_tgt_pc = sqrtf(diff_tgt_pc_x * diff_tgt_pc_x + diff_tgt_pc_y * diff_tgt_pc_y);
        float a_x = diff_pe_pc_x / len_diff_pe_pc;
        float a_y = diff_pe_pc_y / len_diff_pe_pc;
        float b_x = diff_tgt_pc_x / len_diff_tgt_pc;
        float b_y = diff_tgt_pc_y / len_diff_tgt_pc;
        float a_dot_b = a_x * b_x + a_y * b_y;
        if (a_dot_b > 1.f)
          a_dot_b = 1.f;
        else if (a_dot_b < -1.f)
          a_dot_b = -1.f;
        angle = acosf(a_dot_b) * (180.f / PI);
        // Determine angle direction
        float direction = a_x * b_y - a_y * b_x;
        if (direction < 0.f)
          angle = -angle;
        // Make the result look more natural (these checks may be omitted)
        if (angle > 30.f)
          angle = 30.f;
        else if (angle < -30.f)
          angle = -30.f;
        // Save angle
        angle_out[iter - 1] = angle;
        for (int i = 0; i < NUM_JOINTS; i++) 
        {
          if(i < NUM_JOINTS - 1)
          {
            angle_out[i+1] += angle_out[i];
          }
        }
      }
    }

    angles[idx * NUM_JOINTS + 0] = angle_out[0];
    angles[idx * NUM_JOINTS + 1] = angle_out[1];
    angles[idx * NUM_JOINTS + 2] = angle_out[2];
  }
}

int main(int argc, char* argv[])
{
  if(argc != 3)
  {
    std::cerr << "Usage: ./invkin <input file coefficients> <iterations>" << std::endl;
    exit(EXIT_FAILURE);
  }

  float* xTarget_in_h;
  float* yTarget_in_h;
  float* angle_out_h;
  float* angle_out_cpu;

  int data_size = 0;

  // process the files
  std::ifstream coordinate_in_file (argv[1]);
  const int iteration = atoi(argv[2]);

  if(coordinate_in_file.is_open())
  {
    coordinate_in_file >> data_size;
    std::cout << "# Data Size = " << data_size << std::endl;
  }

  // allocate the memory
  xTarget_in_h = new (std::nothrow) float[data_size];
  if(xTarget_in_h == NULL)
  {
    std::cerr << "Memory allocation fails!!!" << std::endl;
    exit(EXIT_FAILURE);  
  }
  yTarget_in_h = new (std::nothrow) float[data_size];
  if(yTarget_in_h == NULL)
  {
    std::cerr << "Memory allocation fails!!!" << std::endl;
    exit(EXIT_FAILURE);  
  }
  angle_out_h = new (std::nothrow) float[data_size*NUM_JOINTS];
  if(angle_out_h == NULL)
  {
    std::cerr << "Memory allocation fails!!!" << std::endl;
    exit(EXIT_FAILURE);  
  }

  angle_out_cpu = new (std::nothrow) float[data_size*NUM_JOINTS];
  if(angle_out_cpu == NULL)
  {
    std::cerr << "Memory allocation fails!!!" << std::endl;
    exit(EXIT_FAILURE);  
  }


  // add data to the arrays
  float xTarget_tmp, yTarget_tmp;
  int coeff_index = 0;
  while(coeff_index < data_size)
  {  
    coordinate_in_file >> xTarget_tmp >> yTarget_tmp;

    for(int i = 0; i < NUM_JOINTS ; i++)
    {
      angle_out_h[coeff_index * NUM_JOINTS + i] = 0.0;
    }

    xTarget_in_h[coeff_index] = xTarget_tmp;
    yTarget_in_h[coeff_index++] = yTarget_tmp;
  }


  std::cout << "# Coordinates are read from file..." << std::endl;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *xTarget_in_d = sycl::malloc_device<float>(data_size, q);
  float *yTarget_in_d = sycl::malloc_device<float>(data_size, q);
  float *angle_out_d = sycl::malloc_device<float>(data_size*NUM_JOINTS, q);

  std::cout << "# Memory allocation on GPU is done..." << std::endl;

  q.memcpy(xTarget_in_d, xTarget_in_h, data_size * sizeof(float));
  q.memcpy(yTarget_in_d, yTarget_in_h, data_size * sizeof(float));

  size_t global_work_size = (data_size +  BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

  q.wait();

  std::cout << "# Data are transfered to GPU..." << std::endl;

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < iteration; n++) {
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class inversek>(
        sycl::nd_range<1>(sycl::range<1>(global_work_size), sycl::range<1>(BLOCK_SIZE)),
        [=] (sycl::nd_item<1> item) {

        int idx = item.get_global_id(0);

        if(idx < data_size)
        {  
          float angle_out[NUM_JOINTS];
          float curr_xTargetIn = xTarget_in_d[idx];
          float curr_yTargetIn = yTarget_in_d[idx];

          for(int i = 0; i < NUM_JOINTS; i++)
          {
            angle_out[i] = 0.0;
          }

          float angle;
          // Initialize x and y data
          float xData[NUM_JOINTS_P1];
          float yData[NUM_JOINTS_P1];

          for (int i = 0 ; i < NUM_JOINTS_P1; i++)
          {
            xData[i] = i;
            yData[i] = 0.f;
          }

          for(int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++)
          {
            for (int iter = NUM_JOINTS; iter > 0; iter--)
            {
              float pe_x = xData[NUM_JOINTS];
              float pe_y = yData[NUM_JOINTS];
              float pc_x = xData[iter-1];
              float pc_y = yData[iter-1];
              float diff_pe_pc_x = pe_x - pc_x;
              float diff_pe_pc_y = pe_y - pc_y;
              float diff_tgt_pc_x = curr_xTargetIn - pc_x;
              float diff_tgt_pc_y = curr_yTargetIn - pc_y;
              float len_diff_pe_pc = sycl::sqrt(diff_pe_pc_x * diff_pe_pc_x + diff_pe_pc_y * diff_pe_pc_y);
              float len_diff_tgt_pc = sycl::sqrt(diff_tgt_pc_x * diff_tgt_pc_x + diff_tgt_pc_y * diff_tgt_pc_y);
              float a_x = diff_pe_pc_x / len_diff_pe_pc;
              float a_y = diff_pe_pc_y / len_diff_pe_pc;
              float b_x = diff_tgt_pc_x / len_diff_tgt_pc;
              float b_y = diff_tgt_pc_y / len_diff_tgt_pc;
              float a_dot_b = a_x * b_x + a_y * b_y;
              if (a_dot_b > 1.f)
                a_dot_b = 1.f;
              else if (a_dot_b < -1.f)
                a_dot_b = -1.f;
              angle = sycl::acos(a_dot_b) * (180.f / PI);
              // Determine angle direction
              float direction = a_x * b_y - a_y * b_x;
              if (direction < 0.f)
                angle = -angle;
              // Make the result look more natural (these checks may be omitted)
              if (angle > 30.f)
                angle = 30.f;
              else if (angle < -30.f)
                angle = -30.f;
              // Save angle
              angle_out[iter - 1] = angle;
              for (int i = 0; i < NUM_JOINTS; i++) 
              {
                if(i < NUM_JOINTS - 1)
                {
                  angle_out[i+1] += angle_out[i];
                }
              }
            }
          }

          angle_out_d[idx * NUM_JOINTS + 0] = angle_out[0];
          angle_out_d[idx * NUM_JOINTS + 1] = angle_out[1];
          angle_out_d[idx * NUM_JOINTS + 2] = angle_out[2];
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-3f) / iteration << " (us)\n";

  q.memcpy(angle_out_h, angle_out_d, data_size * NUM_JOINTS * sizeof(float)).wait();

  // CPU
  invkin_cpu(xTarget_in_h, yTarget_in_h, angle_out_cpu, data_size);

  // Check
  int error = 0;
  for(int i = 0; i < data_size; i++)
  {
    for(int j = 0 ; j < NUM_JOINTS; j++)
    {
      if ( fabsf(angle_out_h[i * NUM_JOINTS + j] - angle_out_cpu[i * NUM_JOINTS + j]) > 1e-3 ) {
        //printf("%f %f\n", angle_out_h[i * NUM_JOINTS + j],  angle_out_cpu[i * NUM_JOINTS + j]);
        error++;
        break;
      }
    } 
  }

  // close files
  coordinate_in_file.close();

  // de-allocate the memory
  delete[] xTarget_in_h;
  delete[] yTarget_in_h;
  delete[] angle_out_h;
  delete[] angle_out_cpu;

  sycl::free(xTarget_in_d, q);
  sycl::free(yTarget_in_d, q);
  sycl::free(angle_out_d, q);

  if (error) 
    std::cout << "FAIL\n";
  else 
    std::cout << "PASS\n";
  if (error) exit(1);

  return 0;
}
