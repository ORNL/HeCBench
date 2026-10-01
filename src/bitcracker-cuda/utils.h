#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>

/* Return codes */
#define BIT_SUCCESS 0
#define BIT_FAILURE 1

/* Sizes (bytes) of the fields parsed out of the input hash */
#define SALT_SIZE 16
#define MAC_SIZE 16
#define NONCE_SIZE 12
#define VMK_SIZE 60

/* Input hash file parsing */
#define HASH_TAG "$bitlocker$"
#define HASH_TAG_LEN (sizeof(HASH_TAG) - 1)
#define INPUT_HASH_SIZE 245
#define NUM_HASH_BLOCKS 0x100000

/* Password reading / layout */
#define MIN_INPUT_PASSWORD_LEN 8
#define FIRST_LENGHT 27
#define SECOND_LENGHT 55
#define PSWD_NUM_CHAR 64
#define PSWD_NUM_UINT32 32

int parse_data(char *input_hash, unsigned char ** salt, unsigned char ** nonce,	unsigned char ** vmk, unsigned char ** mac);
char * strtokm(char *s1, const char *delims);
void * Calloc(size_t len, size_t size);
uint32_t read_password(uint32_t ** buf_i, char ** buf_c, uint32_t max_num_pswd_per_read, FILE *fp);

#endif
