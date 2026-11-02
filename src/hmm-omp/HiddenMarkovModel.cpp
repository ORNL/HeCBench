/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdlib.h>
#include <stdio.h>

// forward declaractions
int initHMM(float *initProb, float *mtState, float *mtObs, const int &nState, const int &nEmit);
int ViterbiCPU(float &viterbiProb,
               int *viterbiPath,
               int *obs, 
               const int &nObs, 
               float *initProb,
               float *mtState, 
               const int &nState,
               float *mtEmit);
int ViterbiGPU(float &viterbiProb,
               int *__restrict__ viterbiPath,
               int *__restrict__ obs, 
               const int nObs, 
               float *__restrict__ initProb,
               float *__restrict__ mtState, 
               const int nState,
               const int nEmit,
               float *__restrict__ mtEmit);



// main function
//*****************************************************************************
int main(int argc, const char **argv)
{
    int nState = 4096; // number of states, must be a multiple of 256
    int nEmit  = 4096; // number of possible observations
    int nDevice = 1;
    
    float *initProb = (float*)malloc(sizeof(float)*nState); // initial probability
    float *mtState  = (float*)malloc(sizeof(float)*nState*nState); // state transition matrix
    float *mtEmit   = (float*)malloc(sizeof(float)*nEmit*nState); // emission matrix
    initHMM(initProb, mtState, mtEmit, nState, nEmit);

    // define observational sequence
    int nObs = 250; // size of observational sequence
    int **obs = (int**)malloc(nDevice*sizeof(int*));
    int **viterbiPathCPU = (int**)malloc(nDevice*sizeof(int*));
    int **viterbiPathGPU = (int**)malloc(nDevice*sizeof(int*));
    float *viterbiProbCPU = (float*)malloc(nDevice*sizeof(float)); 
    float *viterbiProbGPU = (float*)malloc(nDevice*sizeof(float)); 
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        obs[iDevice] = (int*)malloc(sizeof(int)*nObs);
        for (int i = 0; i < nObs; i++)
            obs[iDevice][i] = i % 15;
        viterbiPathCPU[iDevice] = (int*)malloc(sizeof(int)*nObs);
        viterbiPathGPU[iDevice] = (int*)malloc(sizeof(int)*nObs);
    }

    printf("# of states = %d\n# of possible observations = %d \nSize of observational sequence = %d\n\n",
        nState, nEmit, nObs);



    printf("\nCompute Viterbi path on GPU\n");
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        ViterbiGPU(viterbiProbGPU[iDevice], viterbiPathGPU[iDevice], obs[iDevice], nObs, initProb, mtState, nState, nEmit, mtEmit);
    }

    printf("\nCompute Viterbi path on CPU\n");
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        ViterbiCPU(viterbiProbCPU[iDevice], viterbiPathCPU[iDevice], obs[iDevice], nObs, initProb, mtState, nState, mtEmit);
    }
    
    bool pass = true;
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        for (int i = 0; i < nObs; i++)
        {
            if (viterbiPathCPU[iDevice][i] != viterbiPathGPU[iDevice][i]) 
            {
                pass = false;
                break;
            }
        }
    }

    if (pass)
      printf("Success");
    else
      printf("Fail");
    printf("\n");
        
    free(initProb);
    free(mtState);
    free(mtEmit);
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        free(obs[iDevice]);
        free(viterbiPathCPU[iDevice]);
        free(viterbiPathGPU[iDevice]);
    }
    free(obs);
    free(viterbiPathCPU);
    free(viterbiPathGPU);
    free(viterbiProbCPU);
    free(viterbiProbGPU);

    return 0;

}

// initialize initial probability, state transition matrix and emission matrix with random 
// numbers. Note that this does not satisfy the normalization property of the state matrix. 
// However, since the algorithm does not use this property, for testing purpose this is fine.
//*****************************************************************************
int initHMM(float *initProb, float *mtState, float *mtObs, const int &nState, const int &nEmit)
{
    if (nState <= 0 || nEmit <=0) return 0;

    // Initialize initial probability

    for (int i = 0; i < nState; i++) initProb[i] = rand();
    float sum = 0.0;
    for (int i = 0; i < nState; i++) sum += initProb[i];
    for (int i = 0; i < nState; i++) initProb[i] /= sum;

    // Initialize state transition matrix

    for (int i = 0; i < nState; i++) {
        for (int j = 0; j < nState; j++) {
            mtState[i*nState + j] = rand();
            mtState[i*nState + j] /= RAND_MAX;
        }
    }

    // init emission matrix

    for (int i = 0; i < nEmit; i++)
    {
        for (int j = 0; j < nState; j++) 
        {
            mtObs[i*nState + j] = rand();
        }
    }

    // normalize the emission matrix
    for (int j = 0; j < nState; j++) 
    {
        float sum = 0.0;
        for (int i = 0; i < nEmit; i++) sum += mtObs[i*nState + j];
        for (int i = 0; i < nEmit; i++) mtObs[i*nState + j] /= sum;
    }

    return 1;
}
