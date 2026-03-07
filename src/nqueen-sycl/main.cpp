#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define _QUEENS_BLOCK_SIZE_   128
#define _EMPTY_      -1

typedef struct queen_root{
  unsigned int control;
  int8_t board[12];
} QueenRoot;


inline void prefixesHandleSol(QueenRoot *root_prefixes, unsigned int flag,
                              const char *board, int initialDepth, int num_sol)
{
  root_prefixes[num_sol].control = flag;
  for(int i = 0; i<initialDepth;++i)
    root_prefixes[num_sol].board[i] = board[i];
}

inline bool MCstillLegal(const char *board, const int r)
{
  // Check vertical
  for (int i = 0; i < r; ++i)
    if (board[i] == board[r]) return false;
  // Check diagonals
  int ld = board[r];  //left diagonal columns
  int rd = board[r];  // right diagonal columns
  for (int i = r-1; i >= 0; --i) {
    --ld; ++rd;
    if (board[i] == ld || board[i] == rd) return false;
  }
  return true;
}

bool queens_stillLegal(const char *board, const int r)
{
  bool safe = true;
  // Check vertical
  for (int i = 0; i < r; ++i)
    if (board[i] == board[r]) safe = false;
  // Check diagonals
  int ld = board[r];  //left diagonal columns
  int rd = board[r];  // right diagonal columns
  for (int i = r-1; i >= 0; --i) {
    --ld; ++rd;
    if (board[i] == ld || board[i] == rd) safe = false;
  }
  return safe;
}


void BP_queens_root_dfs(
  sycl::nd_item<1> &item,
  int N, unsigned int nPreFixos, int depthPreFixos,
  const QueenRoot *__restrict root_prefixes,
  unsigned long long *__restrict vector_of_tree_size,
  unsigned long long *__restrict sols)
{
  int idx = item.get_global_id(0);
  if (idx < nPreFixos) {
     unsigned int flag = 0;
     unsigned int bit_test = 0;
     char vertice[20];
     int N_l = N;
     int i, depth;
     unsigned long long  qtd_solutions_thread = 0ULL;
     int depthGlobal = depthPreFixos;
     unsigned long long tree_size = 0ULL;

#pragma unroll 2
    for (i = 0; i < N_l; ++i) {
      vertice[i] = _EMPTY_;
    }

    flag = root_prefixes[idx].control;

#pragma unroll 2
    for (i = 0; i < depthGlobal; ++i)
      vertice[i] = root_prefixes[idx].board[i];

    depth = depthGlobal;

    do {
      vertice[depth]++;
      bit_test = 0;
      bit_test |= (1<<vertice[depth]);
      if(vertice[depth] == N_l){
        vertice[depth] = _EMPTY_;
      } else if (!(flag & bit_test ) && queens_stillLegal(vertice, depth)){
        ++tree_size;
        flag |= (1ULL<<vertice[depth]);
        depth++;
        if (depth == N_l) { //sol
          ++qtd_solutions_thread;
        } else continue;
      } else continue;
      depth--;
      flag &= ~(1ULL<<vertice[depth]);
    } while(depth >= depthGlobal);

    sols[idx] = qtd_solutions_thread;
    vector_of_tree_size[idx] = tree_size;
  }//if
}//kernel

unsigned long long BP_queens_prefixes(int size, int initialDepth,
                                      unsigned long long *tree_size,
                                      QueenRoot *root_prefixes)
{
  unsigned int flag = 0;
  int bit_test = 0;
  char vertice[20];
  int i, nivel;
  unsigned long long local_tree = 0ULL;
  unsigned long long num_sol = 0;

  for (i = 0; i < size; ++i) {
    vertice[i] = -1;
  }

  nivel = 0;

  do{

    vertice[nivel]++;
    bit_test = 0;
    bit_test |= (1<<vertice[nivel]);

    if(vertice[nivel] == size){
      vertice[nivel] = _EMPTY_;
    }else if ( MCstillLegal(vertice, nivel) && !(flag &  bit_test ) ){ //is legal

      flag |= (1ULL<<vertice[nivel]);
      nivel++;
      ++local_tree;
      if (nivel == initialDepth){ //handle solution
        prefixesHandleSol(root_prefixes,flag,vertice,initialDepth,num_sol);
        num_sol++;
      }else continue;
    }else continue;

    nivel--;
    flag &= ~(1ULL<<vertice[nivel]);

  }while(nivel >= 0);

  *tree_size = local_tree;

  return num_sol;
}


void nqueens(short size, int initial_depth, unsigned int n_explorers, QueenRoot *root_prefixes_h ,
             unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, const int repeat)
{
  int num_blocks = ceil((double)n_explorers/_QUEENS_BLOCK_SIZE_);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  unsigned long long *vector_of_tree_size_d = sycl::malloc_device<unsigned long long>(n_explorers, q);
  unsigned long long *sols_d = sycl::malloc_device<unsigned long long>(n_explorers, q);
  QueenRoot *root_prefixes_d = sycl::malloc_device<QueenRoot>(n_explorers, q);
  q.memcpy(root_prefixes_d, root_prefixes_h, n_explorers * sizeof(QueenRoot));

  printf("\n### Regular BP-DFS search. ###\n");
  sycl::range<1> gws (num_blocks * _QUEENS_BLOCK_SIZE_);
  sycl::range<1> lws (_QUEENS_BLOCK_SIZE_);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class nqueen>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        BP_queens_root_dfs(item,
                           size,
                           n_explorers,
                           initial_depth,
                           root_prefixes_d,
                           vector_of_tree_size_d,
                           sols_d);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(vector_of_tree_size_h, vector_of_tree_size_d, n_explorers*sizeof(unsigned long long));
  q.memcpy(sols_h, sols_d, n_explorers*sizeof(unsigned long long));
  q.wait();

  sycl::free(vector_of_tree_size_d, q);
  sycl::free(sols_d, q);
  sycl::free(root_prefixes_d, q);
}


int main(int argc, char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <size> <initial depth> <repeat>\n", argv[0]);
    return 1;
  }
  const short size = atoi(argv[1]);  // 15 - 17 for a short run
  const int initialDepth = atoi(argv[2]); // 6 or 7
  const int repeat = atoi(argv[3]); // kernel execution times
  printf("\n### Initial depth: %d - Size: %d:", initialDepth, size);

  unsigned long long tree_size = 0ULL;
  unsigned long long qtd_sols_global = 0ULL;
  unsigned int nMaxPrefixos = 75580635;

  QueenRoot* root_prefixes_h = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixos);
  unsigned long long *vector_of_tree_size_h = (unsigned long long*)malloc(sizeof(unsigned long long)*nMaxPrefixos);
  unsigned long long *solutions_h = (unsigned long long*)malloc(sizeof(unsigned long long)*nMaxPrefixos);

  if (root_prefixes_h == NULL || vector_of_tree_size_h == NULL || solutions_h == NULL) {
    printf("Error: host out of memory\n");
    if (root_prefixes_h) free(root_prefixes_h);
    if (vector_of_tree_size_h) free(vector_of_tree_size_h);
    if (solutions_h) free(solutions_h);
    return 1;
  }

  //initial search, getting the tree root nodes for the gpu;
  unsigned long long n_explorers = BP_queens_prefixes(size, initialDepth, &tree_size, root_prefixes_h);

  //calling the gpu-based search
  nqueens(size, initialDepth, n_explorers, root_prefixes_h, vector_of_tree_size_h, solutions_h, repeat);

  printf("\nTree size: %llu", tree_size );

  for(unsigned long long i = 0; i<n_explorers;++i){
    if(solutions_h[i]>0)
      qtd_sols_global += solutions_h[i];
    if(vector_of_tree_size_h[i]>0)
      tree_size +=vector_of_tree_size_h[i];
  }

  printf("\nNumber of solutions found: %llu \nTree size: %llu\n", qtd_sols_global, tree_size );

  // Initial depth: 7 - Size: 15:
  // Tree size: 2466109
  // Number of solutions found: 2279184
  // Tree size: 171129071
  if (size == 15 && initialDepth == 7) {
    if (qtd_sols_global == 2279184 && tree_size == 171129071)
      printf("PASS\n");
    else
      printf("FAIL\n");
  }

  free(root_prefixes_h);
  free(vector_of_tree_size_h);
  free(solutions_h);
  if (qtd_sols_global != 2279184 && tree_size == 171129071) return 1;
  return 0;
}
