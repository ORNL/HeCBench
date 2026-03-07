#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>

template<typename T>
void verify(const T* ref_out, const T* out, int64_t n)
{
  int error = memcmp(ref_out, out, n * sizeof(T));
  if (error) {
    for (int64_t i = 0; i < n; i++) {
      if (ref_out[i] != out[i]) {
        printf("@%zu: %lf != %lf\n", i, (double)ref_out[i], (double)out[i]);
        break;
      }
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);
}

// bank conflict aware optimization
#define LOG_MEM_BANKS 5
#define OFFSET(n) ((n) >> LOG_MEM_BANKS)

template<typename T, int N>
void runTest (const int64_t n, const int repeat, bool timing = false)
{
  int64_t num_blocks = (n + N - 1) / N;

  int64_t nelems = num_blocks * N; // actual total number of elements

  int64_t bytes = nelems * sizeof(T);

  T *in = (T*) malloc (bytes);
  T *out = (T*) malloc (bytes);
  T *ref_out = (T*) malloc (bytes);

  srand(123);
  for (int64_t n = 0; n < nelems; n++) in[n] = rand() % 5 + 1;

  T *t_in = in;
  T *t_out = ref_out;
  for (int64_t n = 0; n < num_blocks; n++) {
    t_out[0] = 0;
    for (int i = 1; i < N; i++)
      t_out[i] = t_out[i-1] + t_in[i-1];
    t_out += N;
    t_in += N;
  }

  #pragma omp target data map(to: in[0:nelems]) map(alloc: out[0:nelems])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams num_teams(num_blocks) thread_limit(N/2)
      {
        T temp[N];
        #pragma omp parallel
        {
          for (int64_t bid = omp_get_team_num(); bid < num_blocks; bid += omp_get_num_teams())
          {
            T *t_in  = in + bid * N;
            T *t_out = out + bid * N;

            int thid = omp_get_thread_num();
            int offset = 1;

            temp[2*thid]   = t_in[2*thid];
            temp[2*thid+1] = t_in[2*thid+1];

            for (int d = N >> 1; d > 0; d >>= 1)
            {
              #pragma omp barrier
              if (thid < d)
              {
                int ai = offset*(2*thid+1)-1;
                int bi = offset*(2*thid+2)-1;
                temp[bi] += temp[ai];
              }
              offset *= 2;
            }

            if (thid == 0) temp[N-1] = 0; // clear the last elem
            for (int d = 1; d < N; d *= 2) // traverse down
            {
              offset >>= 1;
              #pragma omp barrier
              if (thid < d)
              {
                int ai = offset*(2*thid+1)-1;
                int bi = offset*(2*thid+2)-1;
                T t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
              }
            }
            t_out[2*thid] = temp[2*thid];
            t_out[2*thid+1] = temp[2*thid+1];
	  }
	}
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (timing) {
      printf("Element size in bytes is %zu. Average execution time of scan (w/  bank conflicts): %f (us)\n",
             sizeof(T), (time * 1e-3f) / repeat);
    }
    #pragma omp target update from (out[0:nelems])
    if (!timing) verify(ref_out, out, nelems);

    // bcao
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams num_teams(num_blocks) thread_limit(N/2)
      {
        T temp[2*N];
        #pragma omp parallel
        {
          for (int64_t bid = omp_get_team_num(); bid < num_blocks; bid += omp_get_num_teams())
          {
            T *t_in  = in + bid * N;
            T *t_out = out + bid * N;

            int thid = omp_get_thread_num();
            int a = thid;
            int b = a + (N/2);
            int oa = OFFSET(a);
            int ob = OFFSET(b);

            temp[a + oa] = t_in[a];
            temp[b + ob] = t_in[b];

            int offset = 1;
            for (int d = N >> 1; d > 0; d >>= 1)
            {
              #pragma omp barrier
              if (thid < d)
              {
                int ai = offset*(2*thid+1)-1;
                int bi = offset*(2*thid+2)-1;
                ai += OFFSET(ai);
                bi += OFFSET(bi);
                temp[bi] += temp[ai];
              }
              offset *= 2;
            }

            if (thid == 0) temp[N-1+OFFSET(N-1)] = 0; // clear the last elem
            for (int d = 1; d < N; d *= 2) // traverse down
            {
              offset >>= 1;
              #pragma omp barrier
              if (thid < d)
              {
                int ai = offset*(2*thid+1)-1;
                int bi = offset*(2*thid+2)-1;
                ai += OFFSET(ai);
                bi += OFFSET(bi);
                T t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
              }
            }
            #pragma omp barrier // required

            t_out[a] = temp[a + oa];
            t_out[b] = temp[b + ob];
          }
        }
      }
    }

    end = std::chrono::steady_clock::now();
    auto bcao_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if (timing) {
      printf("Element size in bytes is %zu. Average execution time of scan (w/o bank conflicts): %f (us). ",
             sizeof(T), (bcao_time * 1e-3f) / repeat);
      printf("Reduce the time by %.1f%%\n", (time - bcao_time) * 1.0 / time * 100);
    }

    #pragma omp target update from (out[0:nelems])
    if (!timing) verify(ref_out, out, nelems);
  }
}

template<int N>
void run (const int n, const int repeat) {
  for (int i = 0; i < 2; i++) {
    bool report_timing = i > 0;
    printf("\nThe number of elements to scan in a thread block: %d\n", N);
    runTest< char, N>(n, repeat, report_timing);
    runTest<short, N>(n, repeat, report_timing);
    runTest<  int, N>(n, repeat, report_timing);
    runTest< long, N>(n, repeat, report_timing);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  run< 128>(n, repeat);
  run< 256>(n, repeat);
  run< 512>(n, repeat);
  run<1024>(n, repeat);
  run<2048>(n, repeat);

  return 0;
}
