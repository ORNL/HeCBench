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
#include <cuda.h>
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

#define RK 3  // 3rd order RK
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

__device__
inline void compute_velocity(const float density, const Float3 momentum, Float3* velocity){
  velocity->x = momentum.x / density;
  velocity->y = momentum.y / density;
  velocity->z = momentum.z / density;
}

__device__
inline float compute_speed_sqd(const Float3 velocity){
  return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

__device__
inline float compute_pressure(const float density, const float density_energy, const float speed_sqd){
  return ((float)(GAMMA) - (float)(1.0f))*(density_energy - (float)(0.5f)*density*speed_sqd);
}
// sqrt is a device function
__device__
inline float compute_speed_of_sound(const float density, const float pressure){
  return sqrt((float)(GAMMA)*pressure/density);
}
  __device__ __host__
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


void copy(float* dst, const float* src, const int N){
  cudaMemcpy(dst, src, N*sizeof(float), cudaMemcpyDeviceToDevice);
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


__global__ void initialize_buffer(float *d, const float val, const int nelr)
{
  const int i = (blockDim.x*blockIdx.x + threadIdx.x);
  if (i < nelr) d[i] = val;
}

__global__ void initialize_variables(const int nelr, float* variables, const float* ff_variable)
{
  const int i = (blockDim.x*blockIdx.x + threadIdx.x);
  for(int j = 0; j < NVAR; j++)
    variables[i + j*nelr] = ff_variable[j];
}


__global__ void compute_step_factor(const int nelr, 
    float* variables, 
    float* areas, 
    float* step_factors){

  const int i = (blockDim.x*blockIdx.x + threadIdx.x);
  if( i >= nelr) return;

  float density = variables[i + VAR_DENSITY*nelr];
  Float3 momentum;
  momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
  momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
  momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];

  float density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];

  Float3 velocity;       compute_velocity(density, momentum, &velocity);
  float speed_sqd      = compute_speed_sqd(velocity);

  float pressure       = compute_pressure(density, density_energy, speed_sqd);
  float speed_of_sound = compute_speed_of_sound(density, pressure);
  step_factors[i] = (float)(0.5f) / (sqrt(areas[i]) * (sqrt(speed_sqd) + speed_of_sound));
}


__global__ void 
compute_flux(
    int nelr, 
    int* elements_surrounding_elements,
    float* normals,
    float* variables,
    float* ff_variable,
    float* fluxes,
    Float3* ff_flux_contribution_density_energy,
    Float3* ff_flux_contribution_momentum_x,
    Float3* ff_flux_contribution_momentum_y,
    Float3* ff_flux_contribution_momentum_z){

  const int i = (blockDim.x*blockIdx.x + threadIdx.x);

  if( i >= nelr) return;
  const float smoothing_coefficient = (float)(0.2f);
  int j, nb;
  Float3 normal; 
  float normal_len;
  float factor;

  float density_i = variables[i + VAR_DENSITY*nelr];
  Float3 momentum_i;
  momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
  momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
  momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];

  float density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

  Float3 velocity_i;                     
  compute_velocity(density_i, momentum_i, &velocity_i);
  float speed_sqd_i                          = compute_speed_sqd(velocity_i);
  //float speed_sqd_i;
  //compute_speed_sqd(velocity_i, speed_sqd_i);
  float speed_i                              = sqrt(speed_sqd_i);
  float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
  float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
  Float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
  Float3 flux_contribution_i_density_energy;  
  compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, 
      &flux_contribution_i_momentum_x, &flux_contribution_i_momentum_y, 
      &flux_contribution_i_momentum_z, &flux_contribution_i_density_energy);

  float flux_i_density = (float)(0.0f);
  Float3 flux_i_momentum;
  flux_i_momentum.x = (float)(0.0f);
  flux_i_momentum.y = (float)(0.0f);
  flux_i_momentum.z = (float)(0.0f);
  float flux_i_density_energy = (float)(0.0f);

  Float3 velocity_nb;
  float density_nb, density_energy_nb;
  Float3 momentum_nb;
  Float3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
  Float3 flux_contribution_nb_density_energy;  
  float speed_sqd_nb, speed_of_sound_nb, pressure_nb;

#pragma unroll
  for(j = 0; j < NNB; j++)
  {
    nb = elements_surrounding_elements[i + j*nelr];
    normal.x = normals[i + (j + 0*NNB)*nelr];
    normal.y = normals[i + (j + 1*NNB)*nelr];
    normal.z = normals[i + (j + 2*NNB)*nelr];
    normal_len = sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);

    if(nb >= 0)   // a legitimate neighbor
    {
      density_nb = variables[nb + VAR_DENSITY*nelr];
      momentum_nb.x = variables[nb + (VAR_MOMENTUM+0)*nelr];
      momentum_nb.y = variables[nb + (VAR_MOMENTUM+1)*nelr];
      momentum_nb.z = variables[nb + (VAR_MOMENTUM+2)*nelr];
      density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
      compute_velocity(density_nb, momentum_nb, &velocity_nb);
      speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
      pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
      speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
      compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, 
          &flux_contribution_nb_momentum_x, &flux_contribution_nb_momentum_y, &flux_contribution_nb_momentum_z, 
          &flux_contribution_nb_density_energy);

      // artificial viscosity
      factor = -normal_len*smoothing_coefficient*(float)(0.5f)*(speed_i + sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
      flux_i_density += factor*(density_i-density_nb);
      flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
      flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
      flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
      flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

      // accumulate cell-centered fluxes
      factor = (float)(0.5f)*normal.x;
      flux_i_density += factor*(momentum_nb.x+momentum_i.x);
      flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x+flux_contribution_i_density_energy.x);
      flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x+flux_contribution_i_momentum_x.x);
      flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x+flux_contribution_i_momentum_y.x);
      flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x+flux_contribution_i_momentum_z.x);

      factor = (float)(0.5f)*normal.y;
      flux_i_density += factor*(momentum_nb.y+momentum_i.y);
      flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y+flux_contribution_i_density_energy.y);
      flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y+flux_contribution_i_momentum_x.y);
      flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y+flux_contribution_i_momentum_y.y);
      flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y+flux_contribution_i_momentum_z.y);

      factor = (float)(0.5f)*normal.z;
      flux_i_density += factor*(momentum_nb.z+momentum_i.z);
      flux_i_density_energy += factor*(flux_contribution_nb_density_energy.z+flux_contribution_i_density_energy.z);
      flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.z+flux_contribution_i_momentum_x.z);
      flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.z+flux_contribution_i_momentum_y.z);
      flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.z+flux_contribution_i_momentum_z.z);
    }
    else if(nb == -1)  // a wing boundary
    {
      flux_i_momentum.x += normal.x*pressure_i;
      flux_i_momentum.y += normal.y*pressure_i;
      flux_i_momentum.z += normal.z*pressure_i;
    }
    else if(nb == -2) // a far field boundary
    {
      factor = (float)(0.5f)*normal.x;
      flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
      flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].x+flux_contribution_i_density_energy.x);
      flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].x + flux_contribution_i_momentum_x.x);
      flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].x + flux_contribution_i_momentum_y.x);
      flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].x + flux_contribution_i_momentum_z.x);

      factor = (float)(0.5f)*normal.y;
      flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
      flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].y+flux_contribution_i_density_energy.y);
      flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].y + flux_contribution_i_momentum_x.y);
      flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].y + flux_contribution_i_momentum_y.y);
      flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].y + flux_contribution_i_momentum_z.y);

      factor = (float)(0.5f)*normal.z;
      flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
      flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].z+flux_contribution_i_density_energy.z);
      flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].z + flux_contribution_i_momentum_x.z);
      flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].z + flux_contribution_i_momentum_y.z);
      flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].z + flux_contribution_i_momentum_z.z);

    }
  }

  fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
  fluxes[i + (VAR_MOMENTUM+0)*nelr] = flux_i_momentum.x;
  fluxes[i + (VAR_MOMENTUM+1)*nelr] = flux_i_momentum.y;
  fluxes[i + (VAR_MOMENTUM+2)*nelr] = flux_i_momentum.z;
  fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;

}

__global__ void 
time_step(int j, int nelr, 
    const float* old_variables, 
    float* variables, 
    const float* step_factors, 
    const float* fluxes) {

  const int i = (blockDim.x*blockIdx.x + threadIdx.x);
  if( i >= nelr) return;

  float factor = step_factors[i]/(float)(RK+1-j);

  variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr] + factor*fluxes[i + VAR_DENSITY*nelr];
  variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i + VAR_DENSITY_ENERGY*nelr];
  variables[i + (VAR_MOMENTUM+0)*nelr] = old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
  variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];  
  variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];  
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
  //{
  const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

  h_ff_variable[VAR_DENSITY] = float(1.4);

  float ff_pressure = float(1.0f);
  float ff_speed_of_sound = sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
  float ff_speed = float(ff_mach)*ff_speed_of_sound;

  Float3 ff_velocity;
  ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
  ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
  ff_velocity.z = 0.0f;

  h_ff_variable[VAR_MOMENTUM+0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
  h_ff_variable[VAR_MOMENTUM+1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
  h_ff_variable[VAR_MOMENTUM+2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;

  h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

  Float3 h_ff_momentum;
  h_ff_momentum.x = *(h_ff_variable+VAR_MOMENTUM+0);
  h_ff_momentum.y = *(h_ff_variable+VAR_MOMENTUM+1);
  h_ff_momentum.z = *(h_ff_variable+VAR_MOMENTUM+2);
  Float3 h_ff_flux_contribution_momentum_x;
  Float3 h_ff_flux_contribution_momentum_y;
  Float3 h_ff_flux_contribution_momentum_z;
  Float3 h_ff_flux_contribution_density_energy;
  compute_flux_contribution(h_ff_variable[VAR_DENSITY], 
      h_ff_momentum, 
      h_ff_variable[VAR_DENSITY_ENERGY], 
      ff_pressure,
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

  float* h_variables = new float[nelr*NVAR];

#ifdef DEBUG
  float* h_step_factors = new float[nelr]; 
#endif

  auto offload_start = std::chrono::steady_clock::now();

  float *d_ff_variable;
  Float3 *d_ff_flux_contribution_momentum_x;
  Float3 *d_ff_flux_contribution_momentum_y;
  Float3 *d_ff_flux_contribution_momentum_z;
  Float3 *d_ff_flux_contribution_density_energy;

  cudaMalloc((void**)&d_ff_variable, sizeof(float)*NVAR);
  cudaMemcpy(d_ff_variable, h_ff_variable, sizeof(float)*NVAR, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_ff_flux_contribution_momentum_x, sizeof(Float3));
  cudaMemcpy(d_ff_flux_contribution_momentum_x, &h_ff_flux_contribution_momentum_x, sizeof(Float3), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_ff_flux_contribution_momentum_y, sizeof(Float3));
  cudaMemcpy(d_ff_flux_contribution_momentum_y, &h_ff_flux_contribution_momentum_y, sizeof(Float3), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_ff_flux_contribution_momentum_z, sizeof(Float3));
  cudaMemcpy(d_ff_flux_contribution_momentum_z, &h_ff_flux_contribution_momentum_z, sizeof(Float3), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_ff_flux_contribution_density_energy, sizeof(Float3));
  cudaMemcpy(d_ff_flux_contribution_density_energy, &h_ff_flux_contribution_density_energy, sizeof(Float3), cudaMemcpyHostToDevice);

  float* d_areas;
  cudaMalloc((void**)&d_areas, sizeof(float)*nelr);
  cudaMemcpy(d_areas, h_areas, sizeof(float)*nelr, cudaMemcpyHostToDevice);

  float* d_normals;
  cudaMalloc((void**)&d_normals, sizeof(float)*nelr*NDIM*NNB);
  cudaMemcpy(d_normals, h_normals, sizeof(float)*nelr*NDIM*NNB, cudaMemcpyHostToDevice);

  int* d_elements_surrounding_elements;
  cudaMalloc((void**)&d_elements_surrounding_elements, sizeof(int)*nelr*NNB);
  cudaMemcpy(d_elements_surrounding_elements, h_elements_surrounding_elements, sizeof(int)*nelr*NNB, cudaMemcpyHostToDevice);

  // Create arrays and set initial conditions
  float* d_variables;
  cudaMalloc((void**)&d_variables, sizeof(float)*nelr*NVAR);

  float* d_old_variables;
  cudaMalloc((void**)&d_old_variables, sizeof(float)*nelr*NVAR);

  float* d_fluxes;
  cudaMalloc((void**)&d_fluxes, sizeof(float)*nelr*NVAR);

  float* d_step_factors;
  cudaMalloc((void**)&d_step_factors, sizeof(float)*nelr);

  dim3 gridDim1 ((nelr + BLOCK_SIZE_1 - 1)/BLOCK_SIZE_1);
  dim3 gridDim2 ((nelr + BLOCK_SIZE_2 - 1)/BLOCK_SIZE_2);
  dim3 gridDim3 ((nelr + BLOCK_SIZE_3 - 1)/BLOCK_SIZE_3);
  dim3 gridDim4 ((nelr + BLOCK_SIZE_4 - 1)/BLOCK_SIZE_4);

  cudaDeviceSynchronize();
  auto kernel_start = std::chrono::steady_clock::now();

  initialize_variables<<<gridDim1, BLOCK_SIZE_1>>>(nelr, d_variables, d_ff_variable);
  initialize_variables<<<gridDim1, BLOCK_SIZE_1>>>(nelr, d_old_variables, d_ff_variable);  
  initialize_variables<<<gridDim1, BLOCK_SIZE_1>>>(nelr, d_fluxes, d_ff_variable);    
  initialize_buffer<<<gridDim1, BLOCK_SIZE_1>>>(d_step_factors, 0, nelr);

  // Begin iterations
  for(int n = 0; n < iterations; n++){
    copy(d_old_variables, d_variables, nelr*NVAR);

    // for the first iteration we compute the time step
    compute_step_factor<<<gridDim2, BLOCK_SIZE_2>>>(nelr, d_variables, d_areas, d_step_factors);

#ifdef DEBUG
    cudaMemcpy(h_step_factors, d_step_factors, sizeof(float)*nelr, cudaMemDeviceToHost);
    for (int i = 0; i < 16; i++) printf("step factor: i=%d %f\n", i, h_step_factors[i]);
#endif
    for(int j = 0; j < RK; j++){
      compute_flux<<<gridDim3, BLOCK_SIZE_3>>>(nelr, d_elements_surrounding_elements, d_normals, 
          d_variables, d_ff_variable, d_fluxes, d_ff_flux_contribution_density_energy, \
          d_ff_flux_contribution_momentum_x, d_ff_flux_contribution_momentum_y, 
          d_ff_flux_contribution_momentum_z);
      time_step<<<gridDim4, BLOCK_SIZE_4>>>(j, nelr, d_old_variables, d_variables, d_step_factors, d_fluxes);
    }
  }

  cudaDeviceSynchronize();
  auto kernel_end = std::chrono::steady_clock::now();

  cudaMemcpy(h_variables, d_variables, sizeof(float)*nelr*NVAR, cudaMemcpyDeviceToHost);

  cudaFree(d_ff_variable);
  cudaFree(d_ff_flux_contribution_momentum_x);
  cudaFree(d_ff_flux_contribution_momentum_y);
  cudaFree(d_ff_flux_contribution_momentum_z);
  cudaFree(d_ff_flux_contribution_density_energy);
  cudaFree(d_areas);
  cudaFree(d_normals);
  cudaFree(d_elements_surrounding_elements);
  cudaFree(d_variables);
  cudaFree(d_old_variables);
  cudaFree(d_fluxes);
  cudaFree(d_step_factors);

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
#ifdef DEBUG
  delete[] h_step_factors;
#endif

  std::cout << "Done..." << std::endl;

  return 0;
}
