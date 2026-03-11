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
#include <sycl/sycl.hpp>

#include "support/setup.h"
#include "kernel.h"
#include "support/common.h"
#include "support/verify.h"


// Params ---------------------------------------------------------------------
struct Params {

    int   n_gpu_threads;
    int   n_gpu_blocks;
    int   n_threads;
    int   n_warmup;
    int   n_reps;
    float alpha;
    int   m;
    int   n;
    int   pad;

    Params(int argc, char **argv) {
        n_gpu_threads = 256;
        n_gpu_blocks  = 8;
        n_threads     = 4;
        n_warmup      = 5;
        n_reps        = 1000;
        alpha         = 0.1;
        m             = 1000;
        n             = 999;
        pad           = 1;
        int opt;
        while((opt = getopt(argc, argv, "hi:g:t:w:r:a:m:n:e:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'i': n_gpu_threads = atoi(optarg); break;
            case 'g': n_gpu_blocks  = atoi(optarg); break;
            case 't': n_threads     = atoi(optarg); break;
            case 'w': n_warmup      = atoi(optarg); break;
            case 'r': n_reps        = atoi(optarg); break;
            case 'a': alpha         = atof(optarg); break;
            case 'm': m             = atoi(optarg); break;
            case 'n': n             = atoi(optarg); break;
            case 'e': pad           = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        if(alpha == 0.0) {
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        } else if(alpha == 1.0) {
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else if(alpha > 0.0 && alpha < 1.0) {
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else {
#ifdef DYNAMIC_PARTITION
            assert((n_gpu_threads > 0 && n_gpu_blocks > 0 || n_threads > 0) && "Invalid # of host + device workers!");
#else
            assert(0 && "Illegal value for -a");
#endif
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./pad [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -i <I>    # of device threads per block (default=256)"
                "\n    -g <G>    # of device blocks (default=8)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of input elements to process on host (default=0.1)"
#ifdef DYNAMIC_PARTITION
                "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
#else
                "\n              NOTE: <A> must be between 0.0 and 1.0"
#endif
                "\n"
                "\nBenchmark-specific options:"
                "\n    -m <M>    # of rows (default=1000)"
                "\n    -n <N>    # of columns (default=999)"
                "\n    -e <B>    # of extra columns to pad (default=1)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input(DATA_TYPE *input, const Params &p) {

    // Initialize the host input vectors
    srand(123);
    for(int i = 0; i < p.m; i++) {
        for(int j = 0; j < p.n; j++) {
            input[i * p.n + j] = (DATA_TYPE)rand() / RAND_MAX;
        }
    }
    for(int i = p.n * p.m; i < (p.n + p.pad) * p.m; i++) {
        input[i] = 0.0f;
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);

    // Allocate
    const int in_size     = p.m * (p.n + p.pad);
    const int n_tasks     = divceil(in_size, p.n_gpu_threads * REGS);
    const int n_tasks_cpu = n_tasks * p.alpha;
    const int n_tasks_gpu = n_tasks - n_tasks_cpu;
    const int n_flags     = n_tasks + 1;

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

#ifdef DYNAMIC_PARTITION
    DATA_TYPE * h_in_out = sycl::malloc_shared<DATA_TYPE>(in_size, q);
    DATA_TYPE * d_in_out = h_in_out;
    std::atomic_int *h_flags = sycl::malloc_shared<std::atomic_int>(n_flags, q);
    std::atomic_int *d_flags  = h_flags;
    std::atomic_int * worklist = sycl::malloc_shared<std::atomic_int>(1, q);
#else
    DATA_TYPE *    h_in_out = (DATA_TYPE *)malloc(in_size * sizeof(DATA_TYPE));
    DATA_TYPE *    d_in_out = sycl::malloc_device<DATA_TYPE>(n_tasks_gpu * p.n_gpu_threads * REGS, q);
    std::atomic_int *h_flags = (std::atomic_int *)malloc(n_flags * sizeof(std::atomic_int));
    int* d_flags = sycl::malloc_device<int>(n_flags, q);
    ALLOC_ERR(h_in_out, h_flags);
#endif
    DATA_TYPE *h_in_backup = (DATA_TYPE *)malloc(in_size * sizeof(DATA_TYPE));
    ALLOC_ERR(h_in_backup);

    // Initialize
    const int max_gpu_threads = 256; // 1024
    read_input(h_in_out, p);
    memset(h_flags, 0, n_flags * sizeof(std::atomic_int));
#ifdef DYNAMIC_PARTITION
    h_flags[0].store(1);
#else
    h_flags[0]           = 1;
    h_flags[n_tasks_cpu] = 1;
#endif
    memcpy(h_in_backup, h_in_out, in_size * sizeof(DATA_TYPE)); // Backup for reuse across iterations

#ifndef DYNAMIC_PARTITION
    // Copy to device
    q.memcpy(d_in_out, h_in_out, n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(DATA_TYPE));
    q.memcpy(d_flags, h_flags, n_flags * sizeof(int));
    q.wait();
#endif

    auto start = std::chrono::steady_clock::now();

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memcpy(h_in_out, h_in_backup, in_size * sizeof(DATA_TYPE));
        memset(h_flags, 0, n_flags * sizeof(std::atomic_int));
#ifdef DYNAMIC_PARTITION
        h_flags[0].store(1);
        if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
            worklist[0].store(0);
        }
#else
        h_flags[0]           = 1;
        h_flags[n_tasks_cpu] = 1;
        q.memcpy(d_in_out, h_in_out, n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(DATA_TYPE)).wait();
        q.memcpy(d_flags, h_flags, n_flags * sizeof(int)).wait();
#endif

        // Kernel launch
        if(p.n_gpu_blocks > 0) {
            assert(p.n_gpu_threads <= max_gpu_threads &&
                "The thread block size is greater than the maximum thread block size that can be used on this device");
            call_Padding_kernel(q, p.n_gpu_blocks, p.n_gpu_threads, p.n, p.m, p.pad, n_tasks, p.alpha,
                d_in_out, d_in_out, (int*)d_flags
#ifdef DYNAMIC_PARTITION
                , 1, (int*)worklist
#endif
                );
        }

        // Launch CPU threads
        std::thread main_thread(
            run_cpu_threads, h_in_out, h_in_out, h_flags, p.n, p.m, p.pad, p.n_threads, p.n_gpu_threads, n_tasks, p.alpha
#ifdef DYNAMIC_PARTITION
            , worklist
#endif
            );

        q.wait();
        main_thread.join();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total padding execution time for %d iterations: %f (ms)\n", p.n_reps + p.n_warmup, time * 1e-6f);

#ifndef DYNAMIC_PARTITION
    // Copy back
    if(p.alpha < 1.0) {
        q.memcpy(h_in_out, d_in_out, (n_tasks_gpu * p.n_gpu_threads * REGS > in_size) ?
          in_size * sizeof(DATA_TYPE) : n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(DATA_TYPE)).wait();
    }
#endif

    // Verify answer
    verify(h_in_out, h_in_backup, p.n, p.m, p.n + p.pad);

    // Free memory
#ifdef DYNAMIC_PARTITION
    sycl::free(h_in_out, q);
    sycl::free(h_flags, q);
    sycl::free(worklist, q);
#else
    free(h_in_out);
    free(h_flags);
    sycl::free(d_in_out, q);
    sycl::free(d_flags, q);
#endif
    free(h_in_backup);

    return 0;
}
