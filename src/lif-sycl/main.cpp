#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include <sycl/sycl.hpp>
#include "reference.h"

void lif (
    sycl::nd_item<1> &item,
    size_t numNeurons, int neurons_per_item, float dt,
    const float*__restrict encode_result,
          float*__restrict voltage_array,
          float*__restrict reftime_array,
    float tau_rc, float tau_ref,
    const float*__restrict bias,
    const float*__restrict gain,
          float*__restrict spikes)
{
  size_t i = item.get_global_id(0);
  if( i < numNeurons)
  {
    int neuron_index = i % neurons_per_item;
    int item_index = i / neurons_per_item;

    float voltage = voltage_array[i];
    float ref_time = reftime_array[i];
    float current = bias[neuron_index] + gain[neuron_index] * encode_result[item_index];
    float dV, spike, mult;

    dV = -sycl::expm1(-dt / tau_rc) * (current - voltage);
    voltage = sycl::fmax(voltage + dV, 0.f);

    ref_time -= dt;

    mult = ref_time;
    mult *= -1.f / dt;
    mult += 1.f;

    mult = sycl::fmin(mult, 1.f);
    mult = sycl::fmax(mult, 0.f);

    voltage *= mult;

    if(voltage > 1.f){
      spike = 1.f / dt;
      ref_time = tau_ref + dt * (1.f - (voltage - 1.f) / dV);
      voltage = 0.f;
    }else{
      spike = 0.f;
    }

    reftime_array[i] = ref_time;
    voltage_array[i] = voltage;
    spikes[i] = spike;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <neurons per item> <num_items> <num_steps>\n", argv[0]);
    return 1;
  }
  const int neurons_per_item = atoi(argv[1]);
  const int num_items = atoi(argv[2]);
  const int num_steps = atoi(argv[3]);

  const size_t num_neurons = (size_t)neurons_per_item * num_items;
  const size_t neurons_size = num_neurons * sizeof(float);
  const size_t items_size = num_items * sizeof(float);
  const size_t neurons_per_item_size = neurons_per_item * sizeof(float);

  float dt = 0.1;    // time step
  float tau_rc = 10; // membrane time constant
  float tau_ref = 2; // refactory time

  float* encode_result = (float*) malloc (items_size);
  float* bias = (float*) malloc (neurons_per_item_size);
  float* gain = (float*) malloc (neurons_per_item_size);

  // test
  float* voltage = (float*) malloc (neurons_size);
  float* reftime = (float*) malloc (neurons_size);
  float* spikes = (float*) malloc (neurons_size);;

  // expected
  float* voltage_host = (float*) malloc (neurons_size);
  float* reftime_host = (float*) malloc (neurons_size);
  float* spikes_host = (float*) malloc (neurons_size);;

  srand(123);
  for (int i = 0; i < num_items; i++) {
    encode_result[i] = rand() / (float)RAND_MAX;
  }
  for (size_t i = 0; i < num_neurons; i++) {
    voltage_host[i] = voltage[i] = 1.f + rand() / (float)RAND_MAX;
    reftime_host[i] = reftime[i] = rand() % 5 / 10.f;
  }
  for (int i = 0; i < neurons_per_item; i++) {
    bias[i] = rand() / (float)RAND_MAX;
    gain[i] = rand() / (float)RAND_MAX + 0.5f;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_encode_result = sycl::malloc_device<float>(num_items, q);
  float *d_bias = sycl::malloc_device<float>(neurons_per_item, q);
  float *d_gain = sycl::malloc_device<float>(neurons_per_item, q);
  float *d_voltage = sycl::malloc_device<float>(num_neurons, q);
  float *d_reftime = sycl::malloc_device<float>(num_neurons, q);
  float *d_spikes = sycl::malloc_device<float>(num_neurons, q);

  q.memcpy(d_encode_result, encode_result, items_size);
  q.memcpy(d_bias, bias, neurons_per_item_size);
  q.memcpy(d_gain, gain, neurons_per_item_size);
  q.memcpy(d_voltage, voltage, neurons_size);
  q.memcpy(d_reftime, reftime, neurons_size);

  sycl::range<1> lws (256);
  sycl::range<1> gws ((num_neurons + 255) / 256 * 256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for(int step = 0; step < num_steps; step++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        lif(item,
            num_neurons,
            neurons_per_item,
            dt,
            d_encode_result,
            d_voltage,
            d_reftime,
            tau_rc,
            tau_ref,
            d_bias,
            d_gain,
            d_spikes);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (elapsed_time * 1e-3) / num_steps);

  q.memcpy(spikes, d_spikes, neurons_size);
  q.memcpy(voltage, d_voltage, neurons_size);
  q.memcpy(reftime, d_reftime, neurons_size);
  q.wait();

  for(int step = 0; step < num_steps; step++) {
    reference(num_neurons,
              neurons_per_item,
              dt,
              encode_result,
              voltage_host,
              reftime_host,
              tau_rc,
              tau_ref,
              bias,
              gain,
              spikes_host);
  }

  size_t num_spikes = 0, num_spikes_host = 0;
  std::vector<float> reftime_on_spike, reftime_on_spike_host;
  for (size_t i = 0; i < num_neurons; i++) {
    if (spikes[i] == 1.f / dt) {
      num_spikes++;
      reftime_on_spike.push_back(reftime[i]);
    }
    if (spikes_host[i] == 1.f / dt) {
      num_spikes_host++;
      reftime_on_spike_host.push_back(reftime_host[i]);
    }
  }
  // spikes may occur at slightly different time steps due to
  // numerical differences on host and device
  printf("Number of spikes on host and device: %zu %zu\n", num_spikes_host, num_spikes);
  num_spikes = std::min(num_spikes, num_spikes_host);
  bool ok = true;
  for (size_t i = 0; i < num_spikes; i++) {
    if (fabsf(reftime_on_spike[i] - reftime_on_spike_host[i]) > 0.1f) {
      printf("%f %f\n", reftime_on_spike[i], reftime_on_spike_host[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_encode_result, q);
  sycl::free(d_voltage, q);
  sycl::free(d_reftime, q);
  sycl::free(d_bias, q);
  sycl::free(d_gain, q);
  sycl::free(d_spikes, q);

  free(encode_result);
  free(voltage);
  free(voltage_host);
  free(reftime);
  free(reftime_host);
  free(bias);
  free(gain);
  free(spikes);
  free(spikes_host);

  if (!ok) return 1;
  return 0;
}
