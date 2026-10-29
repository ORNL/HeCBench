/********************************************************************
  euler3d.cpp
  : parallelized code of CFD

  - original code from the AIAA-2009-4001 by Andrew Corrigan, acorriga@gmu.edu
  - parallelization with OpenCL API has been applied by
  Jianbin Fang - j.fang@tudelft.nl
  Delft University of Technology
  Faculty of Electrical Engineering, Mathematics and Computer Science
  Department of Software Technology
  Parallel and Distributed Systems Group
  on 24/03/2011
 ********************************************************************/

#include <iostream>
#include <fstream>
#include <math.h>
#include <sycl/sycl.hpp>
#include "util.h"

/*
 * Options 
 * 
 */ 
#define GAMMA 1.4f
#define iterations 2000
#ifndef block_length
#define block_length 192
#endif

#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


/* Default block_length is 192. Only warn above the 1024-thread
   launch limit, where kernels fail on common GPU targets. */
#if block_length > 1024
#warning "the kernels may fail to launch on some systems if the block length is too large"
#endif

//self-defined user type
typedef struct{
  float x;
  float y;
  float z;
} Float3;

inline void compute_velocity(const float density, const Float3 momentum, Float3* velocity){
  velocity->x = momentum.x / density;
  velocity->y = momentum.y / density;
  velocity->z = momentum.z / density;
}

inline float compute_speed_sqd(const Float3 velocity){
  return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

inline float compute_pressure(const float density, const float density_energy, const float speed_sqd){
  return ((float)(GAMMA) - (float)(1.0f))*(density_energy - (float)(0.5f)*density*speed_sqd);
}
// sqrt is a device function
inline float compute_speed_of_sound(const float density, const float pressure){
  return sycl::sqrt((float)(GAMMA)*pressure/density);
}
inline void compute_flux_contribution(const float density, 
                                      Float3 momentum,
                                      const float density_energy,
                                      const float pressure,
                                      const Float3 velocity,
                                      Float3* fc_momentum_x,
                                      Float3* fc_momentum_y,
                                      Float3* fc_momentum_z,
                                      Float3* fc_density_energy)
{
  fc_momentum_x->x = velocity.x*momentum.x + pressure;
  fc_momentum_x->y = velocity.x*momentum.y;
  fc_momentum_x->z = velocity.x*momentum.z;


  fc_momentum_y->x = fc_momentum_x->y;
  fc_momentum_y->y = velocity.y*momentum.y + pressure;
  fc_momentum_y->z = velocity.y*momentum.z;

  fc_momentum_z->x = fc_momentum_x->z;
  fc_momentum_z->y = fc_momentum_y->z;
  fc_momentum_z->z = velocity.z*momentum.z + pressure;

  const float de_p = density_energy+pressure;
  fc_density_energy->x = velocity.x*de_p;
  fc_density_energy->y = velocity.y*de_p;
  fc_density_energy->z = velocity.z*de_p;
}


template <typename T>
void copy(sycl::queue &q, T *dst, const T *src, const int N){
  q.memcpy(dst, src, N * sizeof(T));
}

void dump(const float *h_variables, const int nel, const int nelr){
  {
    std::ofstream file("density");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY*nelr] << std::endl;
  }


  {
    std::ofstream file("momentum");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++)
    {
      for(int j = 0; j != NDIM; j++)
        file << h_variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
      file << std::endl;
    }
  }

  {
    std::ofstream file("density_energy");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
  }
}

void initialize_buffer(sycl::queue &q, float *mem_d, const float val, const int number_words) noexcept(false) {
  q.memset(mem_d, val, number_words * sizeof(float));
}

void initialize_variables(sycl::queue &q, const int nelr, float *variables_acc, float *ff_variable_acc) noexcept(false) {

  int work_items = nelr;
  int work_group_size = BLOCK_SIZE_1;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class init_vars>(
      sycl::nd_range<1>(sycl::range<1>(work_items),
                        sycl::range<1>(work_group_size)),
      [=] (sycl::nd_item<1> item) {
      #include "kernel_initialize_variables.sycl"
    });
  });
}

void compute_step_factor(sycl::queue &q,
                         const int nelr,
                         float *variables_acc,
                         float *areas_acc,
                         float *step_factors_acc)
{
  int work_items = nelr;
  int work_group_size = BLOCK_SIZE_2;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class compute_step_factor>(
      sycl::nd_range<1>(sycl::range<1>(work_items),
                        sycl::range<1>(work_group_size)),
      [=] (sycl::nd_item<1> item) {
      #include "kernel_compute_step_factor.sycl"
    });
  });
}

void compute_flux(sycl::queue &q,
                  const int nelr,
                  int *elements_surrounding_elements_acc,
                  float *normals_acc,
                  float *variables_acc,
                  float *ff_variable_acc,
                  float *fluxes_acc,
                  Float3 *ff_flux_contribution_density_energy_acc,
                  Float3 *ff_flux_contribution_momentum_x_acc,
                  Float3 *ff_flux_contribution_momentum_y_acc,
                  Float3 *ff_flux_contribution_momentum_z_acc)
{
  int work_items = nelr;
  int work_group_size = BLOCK_SIZE_3;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class compute_flux>(
      sycl::nd_range<1>(sycl::range<1>(work_items),
                        sycl::range<1>(work_group_size)),
      [=] (sycl::nd_item<1> item) {
      #include "kernel_compute_flux.sycl"
    });
  });
}

void time_step(sycl::queue &q,
               const int j,
               const int nelr,
               float *old_variables_acc,
               float *variables_acc,
               float *step_factors_acc,
               float *fluxes_acc)
{
  int work_items = nelr;
  int work_group_size = BLOCK_SIZE_4;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class compute_time_step>(
      sycl::nd_range<1>(sycl::range<1>(work_items),
                        sycl::range<1>(work_group_size)),
      [=] (sycl::nd_item<1> item) {
      #include "kernel_time_step.sycl"
    });
  });
}

/*
 * Main function
 */
int main(int argc, char** argv){
  printf("WG size of kernel:initialize = %d\nWG size of kernel:compute_step_factor = %d\nWG size of kernel:compute_flux = %d\nWG size of kernel:time_step = %d\n", BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3, BLOCK_SIZE_4);

  if (argc < 2){
    std::cout << "Please specify data file name" << std::endl;
    return 0;
  }
  const char* data_file_name = argv[1];
  float h_ff_variable[NVAR];

  std::ifstream file(data_file_name, std::ifstream::in);
  if(!file.good()){
    std::cerr << "Failed to find/open data file. Exit" << std::endl;
    return 1;
  }

  // set far field conditions and load them into constant memory on the gpu
  const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

  h_ff_variable[VAR_DENSITY] = float(1.4);

  float ff_pressure = float(1.0f);
  float ff_speed_of_sound = std::sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
  float ff_speed = float(ff_mach)*ff_speed_of_sound;

  Float3 ff_velocity;
  ff_velocity.x = ff_speed*float(std::cos((float)angle_of_attack));
  ff_velocity.y = ff_speed*float(std::sin((float)angle_of_attack));
  ff_velocity.z = 0.0f;

  h_ff_variable[VAR_MOMENTUM+0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
  h_ff_variable[VAR_MOMENTUM+1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
  h_ff_variable[VAR_MOMENTUM+2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;

  h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) 
	  + (ff_pressure / float(GAMMA-1.0f));

  Float3 h_ff_momentum;
  h_ff_momentum.x = *(h_ff_variable+VAR_MOMENTUM+0);
  h_ff_momentum.y = *(h_ff_variable+VAR_MOMENTUM+1);
  h_ff_momentum.z = *(h_ff_variable+VAR_MOMENTUM+2);
  Float3 h_ff_flux_contribution_momentum_x;
  Float3 h_ff_flux_contribution_momentum_y;
  Float3 h_ff_flux_contribution_momentum_z;
  Float3 h_ff_flux_contribution_density_energy;

  compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum,
                            h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure,
                            ff_velocity, 
                            &h_ff_flux_contribution_momentum_x, 
                            &h_ff_flux_contribution_momentum_y, 
                            &h_ff_flux_contribution_momentum_z,
                            &h_ff_flux_contribution_density_energy);

  int nel;
  int nelr;
  file >> nel;
  nelr = block_length*((nel / block_length )+ std::min(1, nel % block_length));
  std::cout<<"--cambine: nel="<<nel<<", nelr="<<nelr<<std::endl;
  float* h_areas = new float[nelr];
  int* h_elements_surrounding_elements = new int[nelr*NNB];
  float* h_normals = new float[nelr*NDIM*NNB];
  float* h_variables = new float[nelr*NVAR];

  // read in data
  for(int i = 0; i < nel; i++)
  {
    file >> h_areas[i];
    for(int j = 0; j < NNB; j++)
    {
      file >> h_elements_surrounding_elements[i + j*nelr];
      if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
      h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering				

      for(int k = 0; k < NDIM; k++)
      {
        file >> h_normals[i + (j + k*NNB)*nelr];
        h_normals[i + (j + k*NNB)*nelr] = -h_normals[i + (j + k*NNB)*nelr];
      }
    }
  }

  // fill in remaining data
  int last = nel-1;
  for(int i = nel; i < nelr; i++)
  {
    h_areas[i] = h_areas[last];
    for(int j = 0; j < NNB; j++)
    {
      // duplicate the last element
      h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr];	
      for(int k = 0; k < NDIM; k++) h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr];
    }
  }

  auto offload_start = std::chrono::steady_clock::now();

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // copy far field conditions to the gpu
  float *d_ff_variable = sycl::malloc_device<float>(NVAR, q);
  copy(q, d_ff_variable, h_ff_variable, NVAR);
  
  Float3 *d_ff_flux_contribution_momentum_x = sycl::malloc_device<Float3>(1, q);
  copy(q, d_ff_flux_contribution_momentum_x, &h_ff_flux_contribution_momentum_x, 1);

  Float3 *d_ff_flux_contribution_momentum_y = sycl::malloc_device<Float3>(1, q);
  copy(q, d_ff_flux_contribution_momentum_y, &h_ff_flux_contribution_momentum_y, 1);

  Float3 *d_ff_flux_contribution_momentum_z = sycl::malloc_device<Float3>(1, q);
  copy(q, d_ff_flux_contribution_momentum_z, &h_ff_flux_contribution_momentum_z, 1);

  Float3 *d_ff_flux_contribution_density_energy = sycl::malloc_device<Float3>(1, q);
  copy(q, d_ff_flux_contribution_density_energy, &h_ff_flux_contribution_density_energy, 1);

  //upload<float>(q, areas, h_areas, nelr);
  float *d_areas = sycl::malloc_device<float>(nelr, q);
  copy(q, d_areas, h_areas, nelr);

  //upload<int>(q, elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);
  int *d_elements_surrounding_elements = sycl::malloc_device<int>(nelr * NNB, q);
  copy(q, d_elements_surrounding_elements, h_elements_surrounding_elements, nelr * NNB);

  //upload<float>(q, normals, h_normals, nelr*NDIM*NNB);
  float *d_normals = sycl::malloc_device<float>(nelr * NDIM * NNB, q);
  copy(q, d_normals, h_normals, nelr * NDIM * NNB);

  // Create arrays and set initial conditions
  float *d_variables = sycl::malloc_device<float>(nelr * NVAR, q);

  float *d_old_variables = sycl::malloc_device<float>(nelr * NVAR, q);

  float *d_fluxes = sycl::malloc_device<float>(nelr * NVAR, q);

  float *d_step_factors = sycl::malloc_device<float>(nelr, q);

  q.wait();

  auto kernel_start = std::chrono::steady_clock::now();

  initialize_variables(q, nelr, d_variables, d_ff_variable);
  initialize_variables(q, nelr, d_old_variables, d_ff_variable);	
  initialize_variables(q, nelr, d_fluxes, d_ff_variable);		
  initialize_buffer(q, d_step_factors, 0, nelr);

  // these need to be computed the first time in order to compute time step
  //std::cout << "Starting..." << std::endl;

  // Begin iterations
  for(int i = 0; i < iterations; i++){
    copy(q, d_old_variables, d_variables, nelr*NVAR);
    // for the first iteration we compute the time step
    compute_step_factor(q, nelr, d_variables, d_areas, d_step_factors);
    for(int j = 0; j < RK; j++){
      compute_flux(q, nelr, d_elements_surrounding_elements, d_normals, 
                   d_variables, d_ff_variable, d_fluxes,
                   d_ff_flux_contribution_density_energy, \
                   d_ff_flux_contribution_momentum_x,
                   d_ff_flux_contribution_momentum_y, 
                   d_ff_flux_contribution_momentum_z);
      time_step(q, j, nelr, d_old_variables, d_variables, d_step_factors, d_fluxes);
    }
  }

  q.wait();

  auto kernel_end = std::chrono::steady_clock::now();

  copy(q, h_variables, d_variables, nelr * NVAR);
  q.wait();

  sycl::free(d_ff_variable, q);
  sycl::free(d_ff_flux_contribution_momentum_x, q);
  sycl::free(d_ff_flux_contribution_momentum_y, q);
  sycl::free(d_ff_flux_contribution_momentum_z, q);
  sycl::free(d_ff_flux_contribution_density_energy, q);
  sycl::free(d_areas, q);
  sycl::free(d_normals, q);
  sycl::free(d_elements_surrounding_elements, q);
  sycl::free(d_variables, q);
  sycl::free(d_old_variables, q);
  sycl::free(d_fluxes, q);
  sycl::free(d_step_factors, q);

  auto offload_end = std::chrono::steady_clock::now();
  auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count();
  printf("Device offloading time = %lf(s)\n", offload_time * 1e-9);

  auto kernel_time = std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count();
  printf("Total execution time of kernels = %lf(s)\n", kernel_time * 1e-9);

#ifdef OUTPUT
    std::cout << "Saving solution..." << std::endl;
    dump(h_variables, nel, nelr);
#endif

  delete[] h_areas;
  delete[] h_elements_surrounding_elements;
  delete[] h_normals;
  delete[] h_variables;
  std::cout << "Done..." << std::endl;

  return 0;
}
