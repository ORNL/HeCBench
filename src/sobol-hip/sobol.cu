/*
* Portions Copyright (c) 1993-2015 NVIDIA Corporation.  All rights reserved.
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
* Portions Copyright (c) 2009 Mike Giles, Oxford University.  All rights reserved.
* Portions Copyright (c) 2008 Frances Y. Kuo and Stephen Joe.  All rights reserved.
*
* Sobol Quasi-random Number Generator example
*
* Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
* http://people.maths.ox.ac.uk/~gilesm/
*
* and C code developed by Stephen Joe, University of Waikato, New Zealand
* and Frances Kuo, University of New South Wales, Australia
* http://web.maths.unsw.edu.au/~fkuo/sobol/
*
* For theoretical background see:
*
* P. Bratley and B.L. Fox.
* Implementing Sobol's quasirandom sequence generator
* http://portal.acm.org/citation.cfm?id=42288
* ACM Trans. on Math. Software, 14(1):88-100, 1988
*
* S. Joe and F. Kuo.
* Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
* http://portal.acm.org/citation.cfm?id=641879
* ACM Trans. on Math. Software, 29(1):49-57, 2003
*/

#include <math.h>
#include <iostream>
#include <stdexcept>
#include "sobol.h"
#include "sobol_gold.h"
#include "sobol_gpu.h"
#include <cstdlib>

#define L1ERROR_TOLERANCE (1e-6)

void printHelp(int argc, char *argv[])
{
    if (argc > 0)
    {
        std::cout << "\nUsage: " << argv[0] << " <options>\n\n";
    }
    else
    {
        std::cout << "\nUsage: <program name> <options>\n\n";
    }

    std::cout << "\t--vectors=M     specify number of vectors    (required)\n";
    std::cout << "\t                The generator will output M vectors\n\n";
    std::cout << "\t--dimensions=N  specify number of dimensions (required)\n";
    std::cout << "\t                Each vector will consist of N components\n\n";
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 4) {
      printf("Usage: %s <number of vectors> <number of dimensions> <repeat>\n", argv[0]);
      return 1;
    }
    // We will generate n_vectors vectors of n_dimensions numbers
    int n_vectors = atoi(argv[1]); //100000;
    int n_dimensions = atoi(argv[2]); //100;
    int repeat = atoi(argv[3]); //100;

    // Allocate memory for the arrays
    std::cout << "Allocating CPU memory..." << std::endl;
    unsigned int *h_directions = 0;
    float        *h_outputCPU  = 0;
    float        *h_outputGPU  = 0;

    try
    {
        h_directions = new unsigned int [n_dimensions * n_directions];
        h_outputCPU  = new float [n_vectors * n_dimensions];
        h_outputGPU  = new float [n_vectors * n_dimensions];
    }
    catch (const std::exception &e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        std::cerr << "Unable to allocate CPU memory (try running with fewer vectors/dimensions)" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Allocating GPU memory..." << std::endl;
    unsigned int *d_directions;
    float        *d_output;

    size_t direction_size = n_dimensions * n_directions * sizeof(unsigned int);
    size_t output_size = n_vectors * n_dimensions * sizeof(float);

    hipMalloc((void **)&d_directions, direction_size);
    hipMalloc((void **)&d_output, output_size);

    // Initialize the direction numbers (done on the host)
    std::cout << "Initializing direction numbers..." << std::endl;
    initSobolDirectionVectors(n_dimensions, h_directions);

    std::cout << "Executing QRNG on GPU..." << std::endl;

    hipMemcpy(d_directions, h_directions, direction_size, hipMemcpyHostToDevice);
    hipDeviceSynchronize();

    double ktime = sobolGPU(repeat, n_vectors, n_dimensions, d_directions, d_output);

    std::cout << "Average kernel execution time: " << (ktime * 1e-9f) / repeat << " (s)\n";

    hipMemcpy(h_outputGPU, d_output, output_size, hipMemcpyDeviceToHost);

    std::cout << std::endl;
    // Execute the QRNG on the host
    std::cout << "Executing QRNG on CPU..." << std::endl;
    sobolCPU(n_vectors, n_dimensions, h_directions, h_outputCPU);

    // Check the results
    std::cout << "Checking results..." << std::endl;
    float l1norm_diff = 0.0F;
    float l1norm_ref  = 0.0F;
    float l1error;

    // Special case if n_vectors is 1, when the vector should be exactly 0
    if (n_vectors == 1)
    {
        for (int d = 0, v = 0 ; d < n_dimensions ; d++)
        {
            float ref = h_outputCPU[d * n_vectors + v];
            l1norm_diff += fabs(h_outputGPU[d * n_vectors + v] - ref);
            l1norm_ref  += fabs(ref);
        }

        // Output the L1-Error
        l1error = l1norm_diff;

        if (l1norm_ref != 0)
        {
            std::cerr << "Error: L1-Norm of the reference is not zero (for single vector), golden generator appears broken\n";
        }
        else
        {
            std::cout << "L1-Error: " << l1error << std::endl;
        }
    }
    else
    {
        for (int d = 0 ; d < n_dimensions ; d++)
        {
            for (int v = 0 ; v < n_vectors ; v++)
            {
                float ref = h_outputCPU[d * n_vectors + v];
                l1norm_diff += fabs(h_outputGPU[d * n_vectors + v] - ref);
                l1norm_ref  += fabs(ref);
            }
        }

        // Output the L1-Error
        l1error = l1norm_diff / l1norm_ref;

        if (l1norm_ref == 0)
        {
            std::cerr << "Error: L1-Norm of the reference is zero, golden generator appears broken\n";
        }
        else
        {
            std::cout << "L1-Error: " << l1error << std::endl;
        }
    }

    // Cleanup and terminate
    std::cout << "Shutting down..." << std::endl;
    delete h_directions;
    delete h_outputCPU;
    delete h_outputGPU;
    hipFree(d_directions);
    hipFree(d_output);

    // Check pass/fail using L1 error
    if (l1error < L1ERROR_TOLERANCE)
      std::cout << "PASS" << std::endl;
    else 
      std::cout << "FAIL" << std::endl;
    if (l1error >= L1ERROR_TOLERANCE) exit(1);

    return 0;
}
