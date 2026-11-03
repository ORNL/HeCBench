#define __syncthreads() item.barrier(sycl::access::fence_space::local_space)

static inline void atomicAdd(int& val, const int delta)
{
  sycl::atomic_ref<int, 
    sycl::memory_order::relaxed, sycl::memory_scope::device, 
    sycl::access::address_space::global_space> ref(val);
  ref.fetch_add(delta);
}

template <typename T>
static inline void atomicMax(T& val, const T delta)
{
  sycl::atomic_ref<T,
    sycl::memory_order::relaxed, sycl::memory_scope::device, 
    sycl::access::address_space::global_space> ref(val);
  ref.fetch_max(delta);
}

unsigned int LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return *seed;
}

void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}

void FSMKernel(
  sycl::nd_item<1> &item,
  const int length,
  const unsigned short *__restrict data,
  int *best,
  unsigned int *__restrict rndstate,
  unsigned char *__restrict bfsm,
  unsigned char *same,
  int *smax,
  int *sbest,
  int *oldmax,
  unsigned char *next)
{
  int i, d, pc, s, bit, id, misses, rnd;
  unsigned long long myresult;
  unsigned char *fsm, state[TABSIZE];

  int lid = item.get_local_id(0);
  int bid = item.get_group(0);

  fsm = &next[lid * (FSMSIZE * 2)];

  if (lid == 0) {
    oldmax[bid] = 0;
    same[bid] = 0;
  }
  __syncthreads();

  id = item.get_global_id(0);
  rndstate[id] = SEED ^ id;
  LCG_random_init(&rndstate[id]);

  // initial population
  for (i = 0; i < FSMSIZE * 2; i++) {
    fsm[i] = LCG_random(rndstate+id) & (FSMSIZE - 1);
  }

  // run generations until cutoff times no improvement
  do {
    // reset miss counter and initial state
    for (i = 0; i < TABSIZE; i++) state[i] = 0;
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
    if (lid == 0) {
      atomicAdd(best[2], 1);  // increment generation count
      smax[bid] = 0;
      sbest[bid] = 0;
    }
    __syncthreads();
    atomicMax(smax[bid], length - misses);
    __syncthreads();
    if (length - misses == smax[bid]) atomicMax(sbest[bid], lid);
    __syncthreads();
    bit = 0;
    if (sbest[bid] == lid) {
      // check if there was an improvement
      same[bid]++;
      if (oldmax[bid] < smax[bid]) {
        oldmax[bid] = smax[bid];
        same[bid] = 0;
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
        if (lid != sbest[bid])
          fsm[i] = (next[i + sbest[bid] * FSMSIZE * 2] ^ rnd) & (FSMSIZE - 1);
      }
    } else {
      // crossover best FSM with random FSMs using 3/4 of bits from best FSM
      for (i = 0; i < FSMSIZE * 2; i++) {
        rnd = LCG_random(rndstate+id) & LCG_random(rndstate+id);
        if (lid != sbest[bid])
          fsm[i] = (fsm[i] & rnd) | (next[i + sbest[bid] * FSMSIZE * 2] & ~rnd);
      }
    }
  } while (same[bid] < CUTOFF);  // end of loop over generations

  // record best result of this block
  if (sbest[bid] == lid) {
    id = bid;
    myresult = length - misses;
    myresult = (myresult << 32) + id;

    atomicMax(*(unsigned long long *)best, myresult);

    for (i = 0; i < FSMSIZE * 2; i++) {
      bfsm[id * (FSMSIZE*2) + i] = fsm[i];
    }
  }
}

void MaxKernel(
  int *__restrict best, 
  const unsigned char *__restrict bfsm)
{
  // copy best FSM state assignment over
  int id = best[0];
  for (int i = 0; i < FSMSIZE * 2; i++) {
    best[i + 3] = bfsm[id * (FSMSIZE*2) + i];
  }
}
