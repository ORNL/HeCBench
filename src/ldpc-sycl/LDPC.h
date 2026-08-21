/*  Copyright (c) 2011-2016, Robert Wang, email: robertwgh (at) gmail.com
  All rights reserved. https://github.com/robertwgh/cuLDPC

  CUDA implementation of LDPC decoding algorithm.
Created:   10/1/2010
Revision:  08/01/2013
/4/20/2016 prepare for release on Github.
*/

#ifndef LDPC_H
#define LDPC_H

#define YES  1
#define NO  0

// LDPC decoder configurations
#define WIMAX  0
#define WIFI  1
#define MODE  WIMAX

// Simulation parameters
#define NUM_SNR 1
static float snr_array[NUM_SNR] = {3.0f};
#define MIN_FER         2000000 //2000000
#define MIN_CODEWORD    2000 // 9000000
#define MAX_ITERATION 10

#define CW 2 // code words per macro codewords
#define MCW 40 // number of macro codewords
#define MAX_SIM 500

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  The following settings are fixed.
//  They don't need to be changed during simulations.
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#if MODE == WIMAX
// WIMAX
#define Z        96 //1024//96
#define NON_EMPTY_ELMENT 7
#define NON_EMPTY_ELMENT_VNP  6
#else 
// 802.11n
#define Z        81
#define NON_EMPTY_ELMENT 8 //the maximum number of non-empty element in a row of H matrix, for 1944 bit 802.11n code, NON_EMPTY_ELMENT=8
#define NON_EMPTY_ELMENT_VNP  11
#endif

#define BLOCK_SIZE_X ((Z + 32 - 1)/ 32 * 32)
#define THREADS_PER_BLOCK  (BLOCK_SIZE_X * CW)

#define BLK_ROW      12
#define BLK_COL      24
#define HALF_BLK_COL  12

#define BLK_INFO    BLK_ROW
#define BLK_CODEWORD  BLK_COL

#define ROW        (Z*BLK_ROW)
#define COL        (Z*BLK_COL)
#define INFO_LEN    (BLK_INFO * Z)
#define CODEWORD_LEN  (BLK_CODEWORD * Z)

// the slots in the H matrix
#define H_MATRIX    288
// the slots in the compact H matrix
#define H_COMPACT1_ROW  BLK_ROW
#define H_COMPACT1_COL  NON_EMPTY_ELMENT
#define H_COMPACT1 (BLK_ROW * NON_EMPTY_ELMENT) //96 // 8*12

#define H_COMPACT2_ROW  BLK_ROW
#define H_COMPACT2_COL  BLK_COL

typedef struct
{
  int bit_error;
  int frame_error;
} error_result;

typedef struct
{
  char x;
  char y;
  char value;
  char valid;
} h_element;

// Extern function and variable definition
void structure_encode (int s [], int code [], int h[BLK_ROW][BLK_COL]);
void info_gen (int info_bin [], long seed);
void modulation (int code [], float trans []);
void awgn (float trans [], float recv [], long seed);
void llr_init (float llr [], float recv []);
int parity_check (float app[]);
error_result error_check (int info[], int hard_decision[]);

// Variable declaration
extern float sigma ;
extern int *info_bin ;
extern FILE * gfp;


#endif
