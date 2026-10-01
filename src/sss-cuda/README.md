DPmixGGM
========

This folder contains source codes for the "GPU-powered Stochastic Shotgun Search for Dirichlet proces mixtures of Gaussian Graphical Models" 
by Chiranjit Mukherjee and Abel Rodriguez. 

The "DPmixGGM_SSS_main.cpp" file contains tuning parameters for the algorithm, as elaborated below:
1. Run the SSS 
2. Run GPU/CPU versions of the SSS by enabling / disabling the macro CUDA.
3. Specify maximum number of mixture components that the model should accommodate (for pre-allocation of memory).
4. Set SSS runtime parameters C, D, R, S, M, g, h, f, t.
5. Set SSS number of chain parameters. User needs to provide at least one initial point.
7. Set hyperparameters of for the prior on (mu, K | G) with N0, DELTA0.

Complie source codes using the "make" command and run with "./main f9_n150_p50" command.

The program expects an input-data file (e.g. f9_n150_p50) in the DATA/ folder and at least one initialization point (e.g. f9_n150_p50_init1). 
The input-data file should specify n and p in the first row and then provide n rows of length p. The initial point data-file should specify n, p
and L of the initial model configuration in the first row and xi-indices of the initial point in the second row. Subsequent L rows specify 
G_l (l=1:L).

A list of highest-score models is stored in folder RES/.

Dependencies
------------
None beyond a C++17 compiler. The uniform random number generator (MT19937) and
the adaptive Gauss-Kronrod quadrature the sampler needs are implemented in
`gsl_compat.h` in this folder.
