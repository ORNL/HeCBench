#include <cmath>
#include <cstdio>
#include <cfloat>   // FLT_MAX
#include <iostream>
#include <sstream>
#include <chrono>
#include <cuda.h>
#include <cstdlib>

using namespace std;

// leftrotate function definition
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))

#define F(x,y,z) ((x & y) | ((~x) & z))
#define G(x,y,z) ((x & z) | ((~z) & y))
#define H(x,y,z) (x ^ y ^ z)
#define I(x,y,z) (y ^ (x | (~z)))

// This version of the round shifts the interpretation of a,b,c,d by one
// and must be called with v/x/y/z in a matching shuffle pattern.
// Every four Rounds, a,b,c,d are back to their original interpretation,
// thogh, so it all works out in the end (we have 64 rounds per block).
#define ROUND_INPLACE_VIA_SHIFT(w, r, k, v, x, y, z, func)       \
{                                                                \
    v += func(x,y,z) + w + k;                                    \
    v = x + LEFTROTATE(v, r);                                    \
}

// This version ignores the mapping of a/b/c/d to v/x/y/z and simply
// uses a temporary variable to keep the interpretation of a/b/c/d
// consistent.  Whether this one or the previous one performs better
// probably depends on the compiler....
#define ROUND_USING_TEMP_VARS(w, r, k, v, x, y, z, func)         \
{                                                                \
    a = a + func(b,c,d) + k + w;                                 \
    unsigned int temp = d;                                       \
    d = c;                                                       \
    c = b;                                                       \
    b = b + LEFTROTATE(a, r);                                    \
    a = temp;                                                    \
}

// Here, we pick which style of ROUND we use.
#define ROUND ROUND_USING_TEMP_VARS
//#define ROUND ROUND_INPLACE_VIA_SHIFT

/// NOTE: this really only allows a length up to 7 bytes, not 8, because
/// we need to start the padding in the first byte following the message,
/// and we only have two words to work with here....
/// It also assumes words[] has all zero bits except the chars of interest.
__host__ __device__
inline void md5_2words(unsigned int *words, unsigned int len,
    unsigned int *digest)
{
  // For any block but the first one, these should be passed in, not
  // initialized, but we are assuming we only operate on a single block.
  unsigned int h0 = 0x67452301;
  unsigned int h1 = 0xefcdab89;
  unsigned int h2 = 0x98badcfe;
  unsigned int h3 = 0x10325476;

  unsigned int a = h0;
  unsigned int b = h1;
  unsigned int c = h2;
  unsigned int d = h3;

  unsigned int WL = len * 8;
  unsigned int W0 = words[0];
  unsigned int W1 = words[1];

  switch (len)
  {
    case 0: W0 |= 0x00000080; break;
    case 1: W0 |= 0x00008000; break;
    case 2: W0 |= 0x00800000; break;
    case 3: W0 |= 0x80000000; break;
    case 4: W1 |= 0x00000080; break;
    case 5: W1 |= 0x00008000; break;
    case 6: W1 |= 0x00800000; break;
    case 7: W1 |= 0x80000000; break;
      //default: printf("ERROR, ONLY SUPPORT UP TO 7 BYTES IN THIS FUNC\n"); break;
  }

  // args: word data, per-round shift amt, constant, 4 vars, function macro
  ROUND(W0,   7, 0xd76aa478, a, b, c, d, F);
  ROUND(W1,  12, 0xe8c7b756, d, a, b, c, F);
  ROUND(0,   17, 0x242070db, c, d, a, b, F);
  ROUND(0,   22, 0xc1bdceee, b, c, d, a, F);
  ROUND(0,    7, 0xf57c0faf, a, b, c, d, F);
  ROUND(0,   12, 0x4787c62a, d, a, b, c, F);
  ROUND(0,   17, 0xa8304613, c, d, a, b, F);
  ROUND(0,   22, 0xfd469501, b, c, d, a, F);
  ROUND(0,    7, 0x698098d8, a, b, c, d, F);
  ROUND(0,   12, 0x8b44f7af, d, a, b, c, F);
  ROUND(0,   17, 0xffff5bb1, c, d, a, b, F);
  ROUND(0,   22, 0x895cd7be, b, c, d, a, F);
  ROUND(0,    7, 0x6b901122, a, b, c, d, F);
  ROUND(0,   12, 0xfd987193, d, a, b, c, F);
  ROUND(WL,  17, 0xa679438e, c, d, a, b, F);
  ROUND(0,   22, 0x49b40821, b, c, d, a, F);

  ROUND(W1,   5, 0xf61e2562, a, b, c, d, G);
  ROUND(0,    9, 0xc040b340, d, a, b, c, G);
  ROUND(0,   14, 0x265e5a51, c, d, a, b, G);
  ROUND(W0,  20, 0xe9b6c7aa, b, c, d, a, G);
  ROUND(0,    5, 0xd62f105d, a, b, c, d, G);
  ROUND(0,    9, 0x02441453, d, a, b, c, G);
  ROUND(0,   14, 0xd8a1e681, c, d, a, b, G);
  ROUND(0,   20, 0xe7d3fbc8, b, c, d, a, G);
  ROUND(0,    5, 0x21e1cde6, a, b, c, d, G);
  ROUND(WL,   9, 0xc33707d6, d, a, b, c, G);
  ROUND(0,   14, 0xf4d50d87, c, d, a, b, G);
  ROUND(0,   20, 0x455a14ed, b, c, d, a, G);
  ROUND(0,    5, 0xa9e3e905, a, b, c, d, G);
  ROUND(0,    9, 0xfcefa3f8, d, a, b, c, G);
  ROUND(0,   14, 0x676f02d9, c, d, a, b, G);
  ROUND(0,   20, 0x8d2a4c8a, b, c, d, a, G);

  ROUND(0,    4, 0xfffa3942, a, b, c, d, H);
  ROUND(0,   11, 0x8771f681, d, a, b, c, H);
  ROUND(0,   16, 0x6d9d6122, c, d, a, b, H);
  ROUND(WL,  23, 0xfde5380c, b, c, d, a, H);
  ROUND(W1,   4, 0xa4beea44, a, b, c, d, H);
  ROUND(0,   11, 0x4bdecfa9, d, a, b, c, H);
  ROUND(0,   16, 0xf6bb4b60, c, d, a, b, H);
  ROUND(0,   23, 0xbebfbc70, b, c, d, a, H);
  ROUND(0,    4, 0x289b7ec6, a, b, c, d, H);
  ROUND(W0,  11, 0xeaa127fa, d, a, b, c, H);
  ROUND(0,   16, 0xd4ef3085, c, d, a, b, H);
  ROUND(0,   23, 0x04881d05, b, c, d, a, H);
  ROUND(0,    4, 0xd9d4d039, a, b, c, d, H);
  ROUND(0,   11, 0xe6db99e5, d, a, b, c, H);
  ROUND(0,   16, 0x1fa27cf8, c, d, a, b, H);
  ROUND(0,   23, 0xc4ac5665, b, c, d, a, H);

  ROUND(W0,   6, 0xf4292244, a, b, c, d, I);
  ROUND(0,   10, 0x432aff97, d, a, b, c, I);
  ROUND(WL,  15, 0xab9423a7, c, d, a, b, I);
  ROUND(0,   21, 0xfc93a039, b, c, d, a, I);
  ROUND(0,    6, 0x655b59c3, a, b, c, d, I);
  ROUND(0,   10, 0x8f0ccc92, d, a, b, c, I);
  ROUND(0,   15, 0xffeff47d, c, d, a, b, I);
  ROUND(W1,  21, 0x85845dd1, b, c, d, a, I);
  ROUND(0,    6, 0x6fa87e4f, a, b, c, d, I);
  ROUND(0,   10, 0xfe2ce6e0, d, a, b, c, I);
  ROUND(0,   15, 0xa3014314, c, d, a, b, I);
  ROUND(0,   21, 0x4e0811a1, b, c, d, a, I);
  ROUND(0,    6, 0xf7537e82, a, b, c, d, I);
  ROUND(0,   10, 0xbd3af235, d, a, b, c, I);
  ROUND(0,   15, 0x2ad7d2bb, c, d, a, b, I);
  ROUND(0,   21, 0xeb86d391, b, c, d, a, I);

  h0 += a;
  h1 += b;
  h2 += c;
  h3 += d;

  // write the final result out
  digest[0] = h0;
  digest[1] = h1;
  digest[2] = h2;
  digest[3] = h3;
}

// ****************************************************************************
// Function:  FindKeyspaceSize
//
// Purpose:
///   Multiply out the byteLength by valsPerByte to find the 
///   total size of the key space, with error checking.
//
// Arguments:
//   byteLength    number of bytes in a key
//   valsPerByte   number of values each byte can take on
//
// Programmer:  Jeremy Meredith
// Creation:    July 23, 2014
//
// Modifications:
// ****************************************************************************
int FindKeyspaceSize(int byteLength, int valsPerByte)
{
  int keyspace = 1;
  for (int i=0; i<byteLength; ++i)
  {
    if (keyspace >= 0x7fffffff / valsPerByte)
    {
      // error, we're about to overflow a signed int
      return -1;
    }
    keyspace *= valsPerByte;
  }
  return keyspace;
}

// ****************************************************************************
// Function:  IndexToKey
//
// Purpose:
///   For a given index in the keyspace, find the actual key string
///   which is at that index.
//
// Arguments:
//   index         index in key space
//   byteLength    number of bytes in a key
//   valsPerByte   number of values each byte can take on
//   vals          output key string
//
// Programmer:  Jeremy Meredith
// Creation:    July 23, 2014
//
// Modifications:
// ****************************************************************************
__host__ __device__
void IndexToKey(unsigned int index, int byteLength, int valsPerByte,
                unsigned char vals[8])
{
  // loop pointlessly unrolled to avoid CUDA compiler complaints
  // about unaligned accesses (!?) on older compute capabilities
  vals[0] = index % valsPerByte;
  index /= valsPerByte;

  vals[1] = index % valsPerByte;
  index /= valsPerByte;

  vals[2] = index % valsPerByte;
  index /= valsPerByte;

  vals[3] = index % valsPerByte;
  index /= valsPerByte;

  vals[4] = index % valsPerByte;
  index /= valsPerByte;

  vals[5] = index % valsPerByte;
  index /= valsPerByte;

  vals[6] = index % valsPerByte;
  index /= valsPerByte;

  vals[7] = index % valsPerByte;
  index /= valsPerByte;
}


// ****************************************************************************
// Function:  AsHex
//
// Purpose:
///   For a given key string, return the raw hex string for its bytes.
//
// Arguments:
//   vals       key string
//   len        length of key string
//
// Programmer:  Jeremy Meredith
// Creation:    July 23, 2014
//
// Modifications:
// ****************************************************************************
std::string AsHex(unsigned char *vals, int len)
{
  ostringstream out;
  char tmp[256];
  for (int i=0; i<len; ++i)
  {
    sprintf(tmp, "%2.2X", vals[i]);
    out << tmp;
  }
  return out.str();
}


// ****************************************************************************
// Function:  FindKeyWithDigest_CPU
//
// Purpose:
///   On the CPU, search the key space to find a key with the given digest.
//
// Arguments:
//   searchDigest    the digest to search for
//   byteLength      number of bytes in a key
//   valsPerByte     number of values each byte can take on
//   foundIndex      output - the index of the found key (if found)
//   foundKey        output - the string of the found key (if found)
//   foundDigest     output - the digest of the found key (if found)
//
// Programmer:  Jeremy Meredith
// Creation:    July 23, 2014
//
// Modifications:
// ****************************************************************************
void FindKeyWithDigest_CPU(
  const unsigned int searchDigest[4],
  const int byteLength,
  const int valsPerByte,
  int *foundIndex,
  unsigned char foundKey[8],
  unsigned int foundDigest[4])
{
  int keyspace = FindKeyspaceSize(byteLength, valsPerByte);
  for (int i=0; i<keyspace; i += valsPerByte)
  {
    unsigned char key[8] = {0,0,0,0,0,0,0,0};
    IndexToKey(i, byteLength, valsPerByte, key);
    for (int j=0; j < valsPerByte; ++j)
    {
      unsigned int digest[4];
      md5_2words((unsigned int*)key, byteLength, digest);
      if (digest[0] == searchDigest[0] &&
          digest[1] == searchDigest[1] &&
          digest[2] == searchDigest[2] &&
          digest[3] == searchDigest[3])
      {
        *foundIndex = i + j;
        foundKey[0] = key[0];
        foundKey[1] = key[1];
        foundKey[2] = key[2];
        foundKey[3] = key[3];
        foundKey[4] = key[4];
        foundKey[5] = key[5];
        foundKey[6] = key[6];
        foundKey[7] = key[7];
        foundDigest[0] = digest[0];
        foundDigest[1] = digest[1];
        foundDigest[2] = digest[2];
        foundDigest[3] = digest[3];
      }
      ++key[0];
    }
  }
}

// ****************************************************************************
// Function:  FindKeyWithDigest_GPU
//
// Purpose:
///   On the GPU, search the key space to find a key with the given digest.
//
// Arguments:
//   searchDigest    the digest to search for
//   byteLength      number of bytes in a key
//   valsPerByte     number of values each byte can take on
//   foundIndex      output - the index of the found key (if found)
//   foundKey        output - the string of the found key (if found)
//   foundDigest     output - the digest of the found key (if found)
//
// Programmer:  Jeremy Meredith
// Creation:    July 23, 2014
//
// Modifications:
// ****************************************************************************
__global__
void md5hash_kernel(
  int *__restrict__ foundIndex,
  unsigned char *__restrict__ foundKey,
  unsigned int *__restrict__ foundDigest,
  int keyspace,
  const int byteLength,
  const int valsPerByte,
  const unsigned int searchDigest0, 
  const unsigned int searchDigest1, 
  const unsigned int searchDigest2, 
  const unsigned int searchDigest3 )
{
  int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  int startindex = threadid * valsPerByte;
  unsigned char key[8] = {0,0,0,0, 0,0,0,0};
  IndexToKey(startindex, byteLength, valsPerByte, key);

  for (int j=0; j < valsPerByte && startindex+j < keyspace; ++j)
  {
    unsigned int digest[4];
    md5_2words((unsigned int*)key, byteLength, digest);
    if (digest[0] == searchDigest0 &&
        digest[1] == searchDigest1 &&
        digest[2] == searchDigest2 &&
        digest[3] == searchDigest3)
    {
      foundIndex[0] = startindex + j;
      foundKey[0] = key[0];
      foundKey[1] = key[1];
      foundKey[2] = key[2];
      foundKey[3] = key[3];
      foundKey[4] = key[4];
      foundKey[5] = key[5];
      foundKey[6] = key[6];
      foundKey[7] = key[7];
      foundDigest[0] = digest[0];
      foundDigest[1] = digest[1];
      foundDigest[2] = digest[2];
      foundDigest[3] = digest[3];
    }
    ++key[0];
  }   
}

void FindKeyWithDigest_GPU(
  const unsigned int searchDigest[4],
  const int byteLength,
  const int valsPerByte,
  int *foundIndex,
  unsigned char foundKey[8],
  unsigned int foundDigest[4])
{
  int *d_foundIndex;
  unsigned char *d_foundKey;
  unsigned int *d_foundDigest; 
  cudaMalloc((void**)&d_foundIndex, sizeof(int));
  cudaMalloc((void**)&d_foundKey, sizeof(unsigned char) * 8);
  cudaMalloc((void**)&d_foundDigest, sizeof(unsigned int) * 4);
  cudaMemcpy(d_foundKey, foundKey, sizeof(unsigned char) * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(d_foundIndex, foundIndex, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_foundDigest, foundDigest, sizeof(unsigned int) * 4, cudaMemcpyHostToDevice);

  //
  // calculate work thread shape
  //
  int keyspace = FindKeyspaceSize(byteLength, valsPerByte);
  size_t nthreads = 256;
  size_t nblocks  = ceil((double(keyspace) / double(valsPerByte)) / double(nthreads));

  unsigned int searchDigest0 = searchDigest[0];
  unsigned int searchDigest1 = searchDigest[1];
  unsigned int searchDigest2 = searchDigest[2];
  unsigned int searchDigest3 = searchDigest[3];

  md5hash_kernel<<< dim3(nblocks), dim3(nthreads) >>>(
      d_foundIndex, d_foundKey, d_foundDigest, 
      keyspace, byteLength, valsPerByte,
      searchDigest0, searchDigest1, searchDigest2, searchDigest3);

  cudaMemcpy(foundKey, d_foundKey, sizeof(unsigned char) * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(foundIndex, d_foundIndex, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(foundDigest, d_foundDigest, sizeof(unsigned int) * 4, cudaMemcpyDeviceToHost);
  cudaFree(d_foundDigest);
  cudaFree(d_foundIndex);
  cudaFree(d_foundKey);
}


// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the MD5 Hash benchmark
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: July 23, 2014
//
// Modifications:
//
// ****************************************************************************

int main(int argc, char** argv) 
{
  if (argc != 3) {
    printf("Usage: %s <offload>  <passes>\n"
        "offload: 1 for target device offload\n"
        "passes: Number of run iterations\n" , argv[0]);
    return 1;
  }

  // cpu when offload = 0 ; gpu when offload != 0
  int offload = atoi(argv[1]);

  // number of iterations
  int passes = atoi(argv[2]);

  // 
  bool verbose = true;

  for (int size = 1; size <= 4; size++) {
    //
    // Determine the shape/size of key space
    //
    const int sizes_byteLength[]  = { 7,  5,  6,  5};
    const int sizes_valsPerByte[] = {10, 35, 25, 70};

    const int byteLength = sizes_byteLength[size-1];
    const int valsPerByte = sizes_valsPerByte[size-1];

    char atts[1024];
    sprintf(atts, "%dx%d", byteLength, valsPerByte);

    if (verbose)
      cout << "Searching keys of length " << byteLength << " bytes "
           << "and " << valsPerByte << " values per byte" << endl;

    const int keyspace = FindKeyspaceSize(byteLength, valsPerByte);
    if (keyspace < 0)
    {
      cerr << "Error: more than 2^31 bits of entropy is unsupported.\n";
      return -1;
    }

    if (byteLength > 7)
    {
      cerr << "Error: more than 7 byte key length is unsupported.\n";
      return -1;
    }

    if (verbose)
      cout << "|keyspace| = " << keyspace << " ("<<int(keyspace/1e6)<<"M)" << endl;

    //
    // Choose a random key from the keyspace, and calculate its hash.
    //
    srandom(12345);

    for (int pass = 0 ; pass < passes ; ++pass)
    {
      int randomIndex = random() % keyspace;;
      unsigned char randomKey[8] = {0,0,0,0, 0,0,0,0};
      unsigned int randomDigest[4];
      IndexToKey(randomIndex, byteLength, valsPerByte, randomKey);
      md5_2words((unsigned int*)randomKey, byteLength, randomDigest);

      if (verbose)
      {
        cout << endl;
        cout << "--- iteration " << pass << " ---" << endl;
        cout << "Looking for random key:" << endl;
        cout << " randomIndex = " << randomIndex << endl;
        cout << " randomKey   = 0x" << AsHex(randomKey, 8/*byteLength*/) << endl;
        cout << " randomDigest= " << AsHex((unsigned char*)randomDigest, 16) << endl;
      }

      //
      // Use the GPU to brute force search the keyspace for this key.
      //
      unsigned int foundDigest[4] = {0,0,0,0};
      int foundIndex = -1;
      unsigned char foundKey[8] = {0,0,0,0, 0,0,0,0};

      auto start = std::chrono::steady_clock::now();
      if (offload == 0)
      {
        FindKeyWithDigest_CPU(randomDigest, byteLength, valsPerByte,
            &foundIndex, foundKey, foundDigest);
      }
      else
      {
        FindKeyWithDigest_GPU(randomDigest, byteLength, valsPerByte,
            &foundIndex, foundKey, foundDigest);
      }
      auto end = std::chrono::steady_clock::now();
      auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

      //
      // Calculate the rate and add it to the results
      //
      double rate = double(keyspace) / (double(t)/1000) / 1.e9;

      if (verbose)
      {
        cout << "time = " << t << " ms, rate = " << rate << " GHash/sec\n";
      }

      //
      // Double check everything matches (index, key, hash).
      //
      if (foundIndex < 0)
      {
        cerr << "\nERROR: could not find a match.\n";
        rate = FLT_MAX;
      }
      else if (foundIndex != randomIndex)
      {
        cerr << "\nERROR: mismatch in key index found.\n";
        rate = FLT_MAX;
      }
      else if (foundKey[0] != randomKey[0] ||
          foundKey[1] != randomKey[1] ||
          foundKey[2] != randomKey[2] ||
          foundKey[3] != randomKey[3] ||
          foundKey[4] != randomKey[4] ||
          foundKey[5] != randomKey[5] ||
          foundKey[6] != randomKey[6] ||
          foundKey[7] != randomKey[7])
      {
        cerr << "\nERROR: mismatch in key value found.\n";
        rate = FLT_MAX;
      }
      else if (foundDigest[0] != randomDigest[0] ||
          foundDigest[1] != randomDigest[1] ||
          foundDigest[2] != randomDigest[2] ||
          foundDigest[3] != randomDigest[3])
      {
        cerr << "\nERROR: mismatch in digest of key.\n";
        rate = FLT_MAX;
      }
      else
      {
        if (verbose)
          cout << endl << "Successfully found match (index, key, hash):" << endl;
      }

      if (verbose)
      {
        cout << " foundIndex  = " << foundIndex << endl;
        cout << " foundKey    = 0x" << AsHex(foundKey, 8/*byteLength*/) << endl;
        cout << " foundDigest = " << AsHex((unsigned char*)foundDigest, 16) << endl;
        cout << endl;
      }
      cout << ((rate == FLT_MAX) ? "FAIL" : "PASS") << endl;
      if (rate == FLT_MAX) exit(1);
    }
  }

  return 0;
}
