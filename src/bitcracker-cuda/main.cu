/*
 * BitCracker: BitLocker password cracking tool, CUDA version.
 * Copyright (C) 2013-2017  Elena Ago <elena dot ago at gmail dot com>
 *              Massimo Bernaschi <massimo dot bernaschi at gmail dot com>
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

#include <chrono>
#include <iostream>
#include "bitcracker.h"
#include "utils.h"

uint32_t max_num_pswd_per_read = 0;

unsigned char * salt;
size_t tot_word_mem = NUM_HASH_BLOCKS * HASH_BLOCK_NUM_UINT32 * sizeof(uint32_t);

void usage(char *name){
  printf("\nUsage: %s -f <hash_file> -d <dictionary_file> ATTACK TYPE <p|r>\n\n"
      "Options:\n\n"
      "  -h"
      "\t\tShow this help\n"
      "  -f"
      "\t\tPath to your input hash file (HashExtractor output)\n"
      "  -d"
      "\t\tPath to dictionary file\n", name);
}

int main (int argc, char **argv)
{
  int opt = 0;
  int pass_batch_size = 60000;
  char * input_hash = NULL;
  char * input_dictionary = NULL;
  unsigned char *nonce;
  unsigned char *vmk;
  unsigned char *mac;
  uint32_t * d_w_words_uint32 = NULL;

  printf("\n---------> BitCracker: BitLocker password cracking tool <---------\n");

  if (argc < 2) {
    printf("Missing argument!\n");
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }

  while (1) {
    opt = getopt(argc, argv, "f:d:b:h");
    if (opt == -1)
      break;

    switch (opt) {
      case 'f':
        if(strlen(optarg) >= INPUT_SIZE)
        {
          fprintf(stderr, "ERROR: Input hash file path is longer than %d\n", INPUT_SIZE);
          exit(EXIT_FAILURE);
        }
        input_hash=(char *)Calloc(INPUT_SIZE, sizeof(char));
        strncpy(input_hash, optarg, strlen(optarg)+1);
        break;

      case 'd':
        if(strlen(optarg) >= INPUT_SIZE)
        {
          fprintf(stderr, "ERROR: Dictionary file path is longer than %d\n", INPUT_SIZE);
          exit(EXIT_FAILURE);
        }
        input_dictionary=(char *)Calloc(INPUT_SIZE, sizeof(char));
        strncpy(input_dictionary, optarg, strlen(optarg)+1);
        break;

      case 'b':
        pass_batch_size = atoi(optarg);
        break;

      case 'h':
        usage(argv[0]);
        exit(EXIT_FAILURE);
        break;

      default:
        exit(EXIT_FAILURE);
    }
  }

  if (optind < argc) {
    printf ("non-option ARGV-elements: ");
    while (optind < argc)
      printf ("%s ", argv[optind++]);
    putchar ('\n');
    exit(EXIT_FAILURE);
  }

  if (input_dictionary == NULL){
    printf("Missing dictionary file!\n");
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_hash == NULL){
    printf("Missing input hash file!\n");
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }

  max_num_pswd_per_read = pass_batch_size;

  printf("\n\n==================================\n");
  printf("Retrieving Info\n==================================\n\n");
  if(parse_data(input_hash, &salt, &nonce, &vmk, &mac) == BIT_FAILURE)
  {
    fprintf(stderr, "Input hash format error... exit!\n");
    goto cleanup;
  }

  if(salt == NULL || nonce == NULL || vmk == NULL || mac == NULL)
  {
    fprintf(stderr, "NULL MAC string error... exit!\n");
    goto cleanup;
  }

  // allocate memory
  CUDA_CHECK( cudaMalloc((void **)&d_w_words_uint32, NUM_HASH_BLOCKS * HASH_BLOCK_NUM_UINT32 * sizeof(uint32_t)) );

  if(evaluate_w_block(salt, d_w_words_uint32) == BIT_FAILURE)
  {
    fprintf(stderr, "Words error... exit!\n");
    goto cleanup;
  }

  std::cout << "================================================\n";
  std::cout << "                  Attack\n";
  std::cout << "================================================\n";

  attack(input_dictionary, d_w_words_uint32, vmk, nonce, mac, pass_batch_size);

cleanup:
  free(input_hash);
  free(input_dictionary);

  if (d_w_words_uint32 != NULL) {
    CUDA_CHECK( cudaFree(d_w_words_uint32) );
  }
  return 0;
}
