__device__
unsigned int LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return *seed;
}

__device__
void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}

__global__
void FSMKernel(
  const int length,
  const unsigned short *__restrict__ data,
  int *best,
  unsigned int *__restrict__ rndstate,
  unsigned char *__restrict__ bfsm,
  unsigned char *same,
  int *smax,
  int *sbest,
  int *oldmax)
{
  int i, d, pc, s, bit, id, misses, rnd;
  unsigned long long myresult;
  unsigned char *fsm, state[TABSIZE];
  __shared__ unsigned char next[FSMSIZE * 2 * POPSIZE];

  fsm = &next[threadIdx.x * (FSMSIZE * 2)];

  if (threadIdx.x == 0) {
    oldmax[blockIdx.x] = 0;
    same[blockIdx.x] = 0;
  }
  __syncthreads();

  id = threadIdx.x + blockIdx.x * blockDim.x;
  rndstate[id] = SEED ^ id;
  LCG_random_init(&rndstate[id]);

  // initial population
  for (i = 0; i < FSMSIZE * 2; i++) {
    fsm[i] = LCG_random(rndstate+id) & (FSMSIZE - 1);
  }

  // run generations until cutoff times no improvement
  do {
    // reset miss counter and initial state
    memset(state, 0, TABSIZE);
    misses = 0;

    // evaluate FSM
#pragma unroll
    for (i = 0; i < length; i++) {
      d = (int)data[i];
      pc = (d >> 1) & (TABSIZE - 1);
      bit = d & 1;
      s = (int)state[pc];
      misses += bit ^ (s & 1);
      state[pc] = fsm[s + s + bit];
    }
    for (; i < length; i++) {
      d = (int)data[i];
      pc = (d >> 1) & (TABSIZE - 1);
      bit = d & 1;
      s = (int)state[pc];
      misses += bit ^ (s & 1);
      state[pc] = fsm[s + s + bit];
    }

    // determine best FSM
    if (threadIdx.x == 0) {
      atomicAdd(&best[2], 1);  // increment generation count
      smax[blockIdx.x] = 0;
      sbest[blockIdx.x] = 0;
    }
    __syncthreads();
    atomicMax(&smax[blockIdx.x], length - misses);
    __syncthreads();
    if (length - misses == smax[blockIdx.x]) atomicMax(&sbest[blockIdx.x], threadIdx.x);
    __syncthreads();
    bit = 0;
    if (sbest[blockIdx.x] == threadIdx.x) {
      // check if there was an improvement
      same[blockIdx.x]++;
      if (oldmax[blockIdx.x] < smax[blockIdx.x]) {
        oldmax[blockIdx.x] = smax[blockIdx.x];
        same[blockIdx.x] = 0;
      }
    } else {
      // select 1/8 of threads for mutation (best FSM does crossover)
      if ((LCG_random(rndstate+id) & 7) == 0) bit = 1;
    }
    __syncthreads();

    if (bit) {
      // mutate best FSM by flipping random bits with 1/4th probability
      for (i = 0; i < FSMSIZE * 2; i++) {
        rnd = LCG_random(rndstate+id) & LCG_random(rndstate+id);
        if (threadIdx.x != sbest[blockIdx.x])
          fsm[i] = (next[i + sbest[blockIdx.x] * FSMSIZE * 2] ^ rnd) & (FSMSIZE - 1);
      }
    } else {
      // crossover best FSM with random FSMs using 3/4 of bits from best FSM
      for (i = 0; i < FSMSIZE * 2; i++) {
        rnd = LCG_random(rndstate+id) & LCG_random(rndstate+id);
        if (threadIdx.x != sbest[blockIdx.x])
          fsm[i] = (fsm[i] & rnd) | (next[i + sbest[blockIdx.x] * FSMSIZE * 2] & ~rnd);
      }
    }
  } while (same[blockIdx.x] < CUTOFF);  // end of loop over generations

  // record best result of this block
  if (sbest[blockIdx.x] == threadIdx.x) {
    id = blockIdx.x;
    myresult = length - misses;
    myresult = (myresult << 32) + id;

    atomicMax((unsigned long long *)best, myresult);

    for (i = 0; i < FSMSIZE * 2; i++) {
      bfsm[id * (FSMSIZE*2) + i] = fsm[i];
    }
  }
}

__global__
void MaxKernel(
  int *__restrict__ best, 
  const unsigned char *__restrict__ bfsm)
{
  // copy best FSM state assignment over
  int id = best[0];
  for (int i = 0; i < FSMSIZE * 2; i++) {
    best[i + 3] = bfsm[id * (FSMSIZE*2) + i];
  }
}

