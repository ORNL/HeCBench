/*
 * BitCracker: BitLocker password cracking tool, HIP version.
 * Copyright (C) 2013-2017  Elena Ago <elena dot ago at gmail dot com>
 *							Massimo Bernaschi <massimo dot bernaschi at gmail dot com>
 * 
 * This file is part of the BitCracker project: https://github.com/e-ago/bitcracker
 * 
 * BitCracker is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * BitCracker is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with BitCracker. If not, see <http://www.gnu.org/licenses/>.
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <ctype.h>
#include <getopt.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "sha256.h"
#include "utils.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define AUTHENTICATOR_LENGTH 16
#define AES_CTX_LENGTH 256
#define FALSE 0
#define TRUE 1
#define IV_SIZE 16
#define VMK_HEADER_SIZE 12
#define VMK_BODY_SIZE 32
#define VMK_FULL_SIZE 44


#define DICT_BUFSIZE	(50*1024*1024)
#define MAX_PLEN 32

#define RECOVERY_KEY_SIZE_CHAR 56
#define RECOVERY_PASS_BLOCKS 8
#define MODE_RECV_PASS	2

#ifndef UINT32_C
#define UINT32_C(c) c ## UL
#endif

#define HASH_SIZE 8
#define ROUND_SHA_NUM 64
#define SINGLE_BLOCK_W_SIZE 64
#define HASH_BLOCK_NUM_UINT32 64
#define PADDING_SIZE 40
#define WORD_SIZE 4
#define INPUT_SIZE 2048
#define FIXED_PART_INPUT_CHAIN_HASH 88
#define MAX_INPUT_PASSWORD_LEN 27

#define BLOCK_UNIT 32
#define HASH_SIZE_STRING 32

#define HIP_THREADS_NO_MAC 1024
#define THREADS_PER_BLOCK 256

#define HIP_CHECK(call) {                                              \
    hipError_t err = call;                                             \
    if( hipSuccess != err) {                                           \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",  \
            __FILE__, __LINE__, hipGetErrorString( err) );             \
        exit(EXIT_FAILURE);                                            \
    } }

#define HIP_CHECK_LAST_ERROR() {                                            \
    if( hipSuccess != hipGetLastError()) {                                  \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
            __FILE__, __LINE__, hipGetErrorString( hipGetLastError() ) );   \
        exit(EXIT_FAILURE);                                                 \
    } }

extern uint32_t max_num_pswd_per_read;
extern unsigned char * salt;

int evaluate_w_block(unsigned char * salt, uint32_t * d_w_words_uint32);
void attack(char *dname, uint32_t * d_w_words_uint32, unsigned char * encryptedVMK, unsigned char * nonce, unsigned char * encryptedMAC, int gridBlocks);
