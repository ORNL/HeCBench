#include <stdio.h>      /* defines printf for tests */
#include <stdlib.h>     /* defines atol and posix_memalign */
#include <string.h>     /* defines memcpy */
#include <chrono>
#include <omp.h>

#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

/*
   -------------------------------------------------------------------------------
   mix -- mix 3 32-bit values reversibly.

   This is reversible, so any information in (a,b,c) before mix() is
   still in (a,b,c) after mix().

   If four pairs of (a,b,c) inputs are run through mix(), or through
   mix() in reverse, there are at least 32 bits of the output that
   are sometimes the same for one pair and different for another pair.
   This was tested for:
 * pairs that differed by one bit, by two bits, in any combination
 of top bits of (a,b,c), or in any combination of bottom bits of
 (a,b,c).
 * "differ" is defined as +, -, ^, or ~^.  For + and -, I transformed
 the output delta to a Gray code (a^(a>>1)) so a string of 1's (as
 is commonly produced by subtraction) look like a single 1-bit
 difference.
 * the base values were pseudorandom, all zero but one bit set, or 
 all zero plus a counter that starts at zero.

 Some k values for my "a-=c; a^=rot(c,k); c+=b;" arrangement that
 satisfy this are
 4  6  8 16 19  4
 9 15  3 18 27 15
 14  9  3  7 17  3
 Well, "9 15 3 18 27 15" didn't quite get 32 bits diffing
 for "differ" defined as + with a one-bit base and a two-bit delta.  I
 used http://burtleburtle.net/bob/hash/avalanche.html to choose 
 the operations, constants, and arrangements of the variables.

 This does not achieve avalanche.  There are input bits of (a,b,c)
 that fail to affect some output bits of (a,b,c), especially of a.  The
 most thoroughly mixed value is c, but it doesn't really even achieve
 avalanche in c.

 This allows some parallelism.  Read-after-writes are good at doubling
 the number of bits affected, so the goal of mixing pulls in the opposite
 direction as the goal of parallelism.  I did what I could.  Rotates
 seem to cost as much as shifts on every machine I could lay my hands
 on, and rotates are much kinder to the top and bottom bits, so I used
 rotates.
 -------------------------------------------------------------------------------
 */
#define mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);  c += b; \
  b -= a;  b ^= rot(a, 6);  a += c; \
  c -= b;  c ^= rot(b, 8);  b += a; \
  a -= c;  a ^= rot(c,16);  c += b; \
  b -= a;  b ^= rot(a,19);  a += c; \
  c -= b;  c ^= rot(b, 4);  b += a; \
}

/*
   -------------------------------------------------------------------------------
   final -- final mixing of 3 32-bit values (a,b,c) into c

   Pairs of (a,b,c) values differing in only a few bits will usually
   produce values of c that look totally different.  This was tested for
 * pairs that differed by one bit, by two bits, in any combination
 of top bits of (a,b,c), or in any combination of bottom bits of
 (a,b,c).
 * "differ" is defined as +, -, ^, or ~^.  For + and -, I transformed
 the output delta to a Gray code (a^(a>>1)) so a string of 1's (as
 is commonly produced by subtraction) look like a single 1-bit
 difference.
 * the base values were pseudorandom, all zero but one bit set, or 
 all zero plus a counter that starts at zero.

 These constants passed:
 14 11 25 16 4 14 24
 12 14 25 16 4 14 24
 and these came close:
 4  8 15 26 3 22 24
 10  8 15 26 3 22 24
 11  8 15 26 3 22 24
 -------------------------------------------------------------------------------
 */
#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c,4);  \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}


#pragma omp declare target
unsigned int mixRemainder(unsigned int a, 
    unsigned int b, 
    unsigned int c, 
    unsigned int k0,
    unsigned int k1,
    unsigned int k2,
    unsigned int length ) 
{
  switch(length)
  {
    case 12: c+=k2; b+=k1; a+=k0; break;
    case 11: c+=k2&0xffffff; b+=k1; a+=k0; break;
    case 10: c+=k2&0xffff; b+=k1; a+=k0; break;
    case 9 : c+=k2&0xff; b+=k1; a+=k0; break;
    case 8 : b+=k1; a+=k0; break;
    case 7 : b+=k1&0xffffff; a+=k0; break;
    case 6 : b+=k1&0xffff; a+=k0; break;
    case 5 : b+=k1&0xff; a+=k0; break;
    case 4 : a+=k0; break;
    case 3 : a+=k0&0xffffff; break;
    case 2 : a+=k0&0xffff; break;
    case 1 : a+=k0&0xff; break;
    case 0 : return c;              /* zero length strings require no mixing */
  }

  final(a,b,c);
  return c;
}
#pragma omp end declare target

unsigned int hashlittle( const void *key, size_t length, unsigned int initval)
{
  unsigned int a,b,c;                                          /* internal state */

  /* Set up the internal state */
  a = b = c = 0xdeadbeef + ((unsigned int)length) + initval;

  const unsigned int *k = (const unsigned int *)key;         /* read 32-bit chunks */

  /*------ all but last block: aligned reads and affect 32 bits of (a,b,c) */
  while (length > 12)
  {
    a += k[0];
    b += k[1];
    c += k[2];
    mix(a,b,c);
    length -= 12;
    k += 3;
  }

  /*----------------------------- handle the last (probably partial) block */
  /* 
   * "k[2]&0xffffff" actually reads beyond the end of the string, but
   * then masks off the part it's not allowed to read.  Because the
   * string is aligned, the masked-off tail is in the same word as the
   * rest of the string.  Every machine with memory protection I've seen
   * does it on word boundaries, so is OK with this.  But VALGRIND will
   * still catch it and complain.  The masking trick does make the hash
   * noticably faster for short strings (like English words).
   */

  switch(length)
  {
    case 12: c+=k[2]; b+=k[1]; a+=k[0]; break;
    case 11: c+=k[2]&0xffffff; b+=k[1]; a+=k[0]; break;
    case 10: c+=k[2]&0xffff; b+=k[1]; a+=k[0]; break;
    case 9 : c+=k[2]&0xff; b+=k[1]; a+=k[0]; break;
    case 8 : b+=k[1]; a+=k[0]; break;
    case 7 : b+=k[1]&0xffffff; a+=k[0]; break;
    case 6 : b+=k[1]&0xffff; a+=k[0]; break;
    case 5 : b+=k[1]&0xff; a+=k[0]; break;
    case 4 : a+=k[0]; break;
    case 3 : a+=k[0]&0xffffff; break;
    case 2 : a+=k[0]&0xffff; break;
    case 1 : a+=k[0]&0xff; break;
    case 0 : return c;              /* zero length strings require no mixing */
  }

  final(a,b,c);
  return c;
}


int main(int argc, char** argv) {

  if (argc != 4) {
    printf("Usage: %s <block size> <number of strings> <repeat>\n", argv[0]);
    return 1;
  }

  int block_size = atoi(argv[1]);  // work group size
  unsigned long N = atol(argv[2]); // total number of strings
  int repeat = atoi(argv[3]);

  // sample gold result
  const char* str = "Four score and seven years ago";
  unsigned int c = hashlittle(str, 30, 1);
  printf("input string: %s hash is %.8x\n", str, c);   /* cd628161 */

  unsigned int *keys = NULL;
  unsigned int *lens = NULL;
  unsigned int *initvals = NULL;
  unsigned int *out = NULL;

  // padded to 64 bytes (16 words)
  posix_memalign((void**)&keys, 1024, sizeof(unsigned int)*N*16);
  posix_memalign((void**)&lens, 1024, sizeof(unsigned int)*N);
  posix_memalign((void**)&initvals, 1024, sizeof(unsigned int)*N);
  posix_memalign((void**)&out, 1024, sizeof(unsigned int)*N);

  // the kernel supports up to 60 bytes
  srand(2);
  char src[64];
  memcpy(src, str, 64);
  for (unsigned long i = 0; i < N; i++) {
    memcpy((unsigned char*)keys+i*16*sizeof(unsigned int), src, 64);
    lens[i] = rand()%61;
    initvals[i] = i%2;
  }

  auto start = std::chrono::steady_clock::now();

  #pragma omp target data map(to: keys[0:N*16], lens[0:N], initvals[0:N]) map(from: out[0:N])
  {
    for (int n = 0; n < repeat; n++) {
      #pragma omp target teams distribute parallel for thread_limit(block_size)
      for (unsigned long id = 0; id < N; id++) {
        unsigned int length = lens[id];
        const unsigned int initval = initvals[id];
        const unsigned int *k = keys+id*16;  // each key has at most 15 words (60 bytes)

        /* Set up the internal state */
        unsigned int a,b,c; 
        unsigned int r0,r1,r2;
        a = b = c = 0xdeadbeef + length + initval;

        /*------ all but last block: aligned reads and affect 32 bits of (a,b,c) */
        while (length > 12) {
          a += k[0];
          b += k[1];
          c += k[2];
          mix(a,b,c);
          length -= 12;
          k += 3;
        }
        r0 = k[0];
        r1 = k[1];
        r2 = k[2];

        /*----------------------------- handle the last (probably partial) block */
        /* 
         * "k[2]&0xffffff" actually reads beyond the end of the string, but
         * then masks off the part it's not allowed to read.  Because the
         * string is aligned, the masked-off tail is in the same word as the
         * rest of the string.  Every machine with memory protection I've seen
         * does it on word boundaries, so is OK with this.  But VALGRIND will
         * still catch it and complain.  The masking trick does make the hash
         * noticably faster for short strings (like English words).
         */
        out[id] = mixRemainder(a, b, c, r0, r1, r2, length);
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time : %f (s)\n", (time * 1e-9f) / repeat);

  printf("Verify the results computed on the device..\n");
  bool error = false;
  for (unsigned long i = 0; i < N; i++) {
    c = hashlittle(&keys[i*16], lens[i], initvals[i]);
    if (out[i] != c) {
      printf("Error: at %lu gpu hash is %.8x  cpu hash is %.8x\n", i, out[i], c);
      error = true;
      break;
    }
  }

  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  free(keys);
  free(lens);
  free(initvals);
  free(out);

  return 0;
}
