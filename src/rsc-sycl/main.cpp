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

#include <string.h>
#include <unistd.h>
#include <thread>
#include <assert.h>
#include <chrono>
#include "kernel.h"

#include "support/setup.h"
#include "support/common.h"
#include "support/verify.h"

// Params ---------------------------------------------------------------------
struct Params {

    int         device;
    int         n_gpu_threads;
    int         n_gpu_blocks;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    const char *file_name;
    int         max_iter;
    int         error_threshold;
    float       convergence_threshold;

    Params(int argc, char **argv) {
        device                = 0;
        n_gpu_threads         = 256;
        n_gpu_blocks          = 64;
        n_threads             = 1;
        n_warmup              = 5;
        n_reps                = 1000;
        file_name             = "input/vectors.csv";
        max_iter              = 2000;
        error_threshold       = 3;
        convergence_threshold = 0.75;
        int opt;
        while((opt = getopt(argc, argv, "hi:g:t:w:r:f:m:e:c:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'i': n_gpu_threads         = atoi(optarg); break;
            case 'g': n_gpu_blocks          = atoi(optarg); break;
            case 't': n_threads             = atoi(optarg); break;
            case 'w': n_warmup              = atoi(optarg); break;
            case 'r': n_reps                = atoi(optarg); break;
            case 'f': file_name             = optarg; break;
            case 'm': max_iter              = atoi(optarg); break;
            case 'e': error_threshold       = atoi(optarg); break;
            case 'c': convergence_threshold = atof(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        assert(n_gpu_threads > 0 && "Invalid # of device threads!");
        assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        assert(n_threads > 0 && "Invalid # of host threads!");
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./rsct [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -i <I>    # of device threads per block (default=256)"
                "\n    -g <G>    # of device blocks (default=64)"
                "\n    -t <T>    # of host threads (default=1)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=1000)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    input file name (default=input/vectors.csv)"
                "\n    -m <M>    maximum # of iterations (default=2000)"
                "\n    -e <E>    error threshold (default=3)"
                "\n    -c <C>    convergence threshold (default=0.75)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
int read_input_size(const Params &p) {
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error opening file!");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    fclose(File);

    return n;
}

void read_input(flowvector *v, int *r, const Params &p) {

    int ic = 0;

    // Open input file
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error opening file!");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    while(fscanf(File, "%d,%d,%d,%d", &v[ic].x, &v[ic].y, &v[ic].vx, &v[ic].vy) == 4) {
        ic++;
        if(ic > n) {
            puts("Error: inconsistent file data!");
            exit(-1);
        }
    }
    if(ic < n) {
        puts("Error: inconsistent file data!");
        exit(-1);
    }

    srand(123);
    for(int i = 0; i < 2 * p.max_iter; i++) {
        r[i] = ((int)rand()) % n;
    }
}

int main(int argc, char **argv) {

    const Params p(argc, argv);

    const int max_gpu_threads = 256;
    assert(p.n_gpu_threads <= max_gpu_threads &&
           "The thread block size is greater than the maximum thread block size that can be used on this device");

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    // Allocate
    int         n_flow_vectors = read_input_size(p);
    int         candidates;
    int         best_model           = -1;
    int         best_outliers        = n_flow_vectors;
    flowvector *h_flow_vector_array  = (flowvector *)malloc(n_flow_vectors * sizeof(flowvector));
    int *       h_random_numbers     = (int *)malloc(2 * p.max_iter * sizeof(int));
    int *       h_model_candidate    = (int *)malloc(p.max_iter * sizeof(int));
    int *       h_outliers_candidate = (int *)malloc(p.max_iter * sizeof(int));
    float *     h_model_param_local  = (float *)malloc(4 * p.max_iter * sizeof(float));
    int *       h_g_out_id           = (int *)malloc(sizeof(int));

    ALLOC_ERR(h_flow_vector_array, h_random_numbers, h_model_candidate, h_outliers_candidate, h_model_param_local,
        h_g_out_id);

    // Initialize flow vector and random numbers
    read_input(h_flow_vector_array, h_random_numbers, p);

    flowvector *d_flow_vector_array = sycl::malloc_device<flowvector>(n_flow_vectors, q);
           int *d_model_candidate = sycl::malloc_device<int>(p.max_iter, q);
           int *d_outliers_candidate = sycl::malloc_device<int>(p.max_iter, q);
         float *d_model_param_local = sycl::malloc_device<float>(4 * p.max_iter, q);
           int *d_g_out_id = sycl::malloc_device<int>(1, q);

    // Copy to device
    q.memcpy(d_flow_vector_array, h_flow_vector_array, n_flow_vectors * sizeof(flowvector));
    q.memcpy(d_model_candidate, h_model_candidate, p.max_iter * sizeof(int));
    q.memcpy(d_outliers_candidate, h_outliers_candidate, p.max_iter * sizeof(int));
    q.memcpy(d_model_param_local, h_model_param_local, 4 * p.max_iter * sizeof(float));
    q.memcpy(d_g_out_id, h_g_out_id, sizeof(int));

    q.wait();

    auto start = std::chrono::steady_clock::now();

    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memset((void *)h_model_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_outliers_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_model_param_local, 0, 4 * p.max_iter * sizeof(float));
        h_g_out_id[0] = 0;

        q.memcpy(d_model_candidate, h_model_candidate, p.max_iter * sizeof(int)).wait();
        q.memcpy(d_outliers_candidate, h_outliers_candidate, p.max_iter * sizeof(int)).wait();
        q.memcpy(d_model_param_local, h_model_param_local, 4 * p.max_iter * sizeof(float)).wait();
        q.memcpy(d_g_out_id, h_g_out_id, sizeof(int)).wait();

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_model_param_local, h_flow_vector_array, n_flow_vectors,
            h_random_numbers, p.max_iter, p.error_threshold, p.convergence_threshold, h_g_out_id, p.n_threads);
        main_thread.join();

        q.memcpy(d_model_param_local, h_model_param_local, 4 * p.max_iter * sizeof(float)).wait();

        // Launch GPU threads
        // Kernel launch

        call_RANSAC_kernel_block(q, p.n_gpu_blocks, p.n_gpu_threads, d_model_param_local, d_flow_vector_array,
            n_flow_vectors, p.max_iter, p.error_threshold, p.convergence_threshold,
            d_g_out_id, d_model_candidate, d_outliers_candidate, 1);

        q.memcpy(&candidates, d_g_out_id, sizeof(int)).wait();
        q.memcpy(h_model_candidate, d_model_candidate, candidates * sizeof(int)).wait();
        q.memcpy(h_outliers_candidate, d_outliers_candidate, candidates * sizeof(int)).wait();

        // Post-processing (chooses the best model among the candidates)
        for(int i = 0; i < candidates; i++) {
            if(h_outliers_candidate[i] < best_outliers) {
                best_outliers = h_outliers_candidate[i];
                best_model    = h_model_candidate[i];
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total task execution time for %d iterations: %f (ms)\n", p.n_reps + p.n_warmup, time * 1e-6f);

    printf("Best model (test) %d\n", best_model);

    // Verify answer
    verify(h_flow_vector_array, n_flow_vectors, h_random_numbers, p.max_iter, p.error_threshold,
           p.convergence_threshold, candidates, best_outliers);

    // Free memory
    free(h_model_candidate);
    free(h_outliers_candidate);
    free(h_model_param_local);
    free(h_g_out_id);
    free(h_flow_vector_array);
    free(h_random_numbers);

    sycl::free(d_model_candidate, q);
    sycl::free(d_outliers_candidate, q);
    sycl::free(d_model_param_local, q);
    sycl::free(d_g_out_id, q);
    sycl::free(d_flow_vector_array, q);

    return 0;
}
