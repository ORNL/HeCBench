#include <unistd.h> // access, F_OK
#include "sgd.h"

using namespace std;

SGDIndex* gen_random_map(SGDIndex size)
{
  srand(123);
  vector<SGDIndex> map(size, 0);
  for(SGDIndex i = 0; i < size; i++) map[i] = i;

  random_device rd;
  mt19937 g(rd());
  shuffle(map.begin(), map.end(), g);

  int*map_ptr = new int[size];
  for(int i = 0;i < size;i++)map_ptr[i] = map[i];

  return map_ptr;
}

SGDIndex* gen_inv_map(SGDIndex*map,int size)
{
  int*inv_map = new int[size];
  for(int i = 0;i < size;i++)inv_map[map[i]] = i;
  return inv_map;
}

struct sort_node_by_p
{
  bool operator() (mf_node const &lhs, mf_node const& rhs)
  {
    return tie(lhs.u, lhs.v) < tie(rhs.u, rhs.v);
  }
};

struct sort_node_by_q
{
  bool operator() (mf_node const &lhs, mf_node const &rhs)
  {
    return tie(lhs.v, lhs.u) < tie(rhs.v, rhs.u);
  }
};

void collect_data(mf_problem *prob, SGDRate& ave, SGDRate& std_dev)
{
  double ex = 0;
  double ex2 = 0;

  for(long long i = 0; i < prob->nnz; i++)
  {
    SGDRate r = prob->R[i].rate;
    ex += (double)r;
    ex2 += (double)r*r;
  }
  ex  = ex/(double)prob->nnz;
  ex2 = ex2/(double)prob->nnz;

  ave = (SGDRate)ex;
  std_dev = (SGDRate)sqrt(ex2-ex*ex);
}

void scale_problem(mf_problem*prob, float scale, long long u_seg, long long v_seg)
{
  if(prob->ux*prob->vy == 1)
  {
    for(long long i = 0;i < prob->nnz; i++)
    {   
      prob->R[i].rate = prob->R[i].rate*scale;
    }
  }
  else
  {
    for(long long i = 0;i < prob->nnz; i++)
    {   
      prob->R[i].rate = prob->R[i].rate*scale;

      long long tmp_u = prob->R[i].u;
      while(tmp_u >= u_seg)tmp_u = tmp_u - u_seg;
      prob->R[i].u = tmp_u;

      long long tmp_v = prob->R[i].v;
      while(tmp_v >= v_seg)tmp_v = tmp_v - v_seg;
      prob->R[i].v = tmp_v;
    }
  }
}

void shuffle_problem(mf_problem*prob, SGDIndex*p_map, SGDIndex*q_map)
{
  for(long long i = 0; i < prob->nnz; i++)
  {
    mf_node &N = prob->R[i];
    N.u = p_map[N.u];
    N.v = q_map[N.v];
  }
}

struct pthread_arg
{
  int thread_id; 
  string path;
  mf_node *R;
  long long offset;
  long long size;
  int max_m;
  int max_n;
};

void *read_problem_thread(void *argument)
{
  pthread_arg *arg = (pthread_arg*)argument;

  FILE*fptr = fopen(arg->path.c_str(), "rb");
  if(fptr == NULL)
  {
    printf("file %s open failed\n", arg->path.c_str());
    exit(0);
  }

  int max_m = -1;
  int max_n = -1;

  for(long long idx = 0;idx < arg->size;idx ++)
  {
    int flag = 0;
    int u,v;
    float r;

    flag += fread(&u, sizeof(int), 1, fptr); 
    flag += fread(&v, sizeof(int), 1, fptr); 
    flag += fread(&r, sizeof(float), 1, fptr); 

    if(flag != 3)break;

    // reject invalid node ids as early as possible; negative ids would
    // later map to an out-of-bounds grid index.
    if(u < 0 || v < 0)
    {
      fprintf(stderr, "invalid record %lld in %s: u=%d, v=%d\n",
          idx, arg->path.c_str(), u, v);
      exit(1);
    }

    if(u + 1 > max_m)max_m = u + 1;
    if(v + 1 > max_n)max_n = v + 1;

    arg->R[idx + arg->offset].u = u;
    arg->R[idx + arg->offset].v = v;
    arg->R[idx + arg->offset].rate = r;

  }
  fclose(fptr);

  arg->max_m = max_m;
  arg->max_n = max_n;
  return NULL;
}

mf_problem read_problem(sycl::queue &q, string path)
{
  printf("read problem called\n");
  struct timespec begin, end;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &begin);

  mf_problem prob;
  prob.m = 1;
  prob.n = 1;
  prob.nnz = 0;
  prob.R = NULL;

  int num_files = 0;
  vector<string> file_names;
  for(int i = 0; i < 80; i++)
  {
    stringstream tmp_name_stream;
    tmp_name_stream << path << i;
    string tmp_name = tmp_name_stream.str();

    if(access(tmp_name.c_str(), F_OK) != -1)file_names.push_back(tmp_name);
  }
  num_files = file_names.size();

  if(num_files <= 0)
  {
    if(path.empty())
    {
      printf("file %s open failed\n", path.c_str());
      exit(0);
      return prob;
    }

    FILE*fptr = fopen(path.c_str(), "rb");
    if(fptr == NULL)
    {
      printf("file %s open failed\n", path.c_str());
      exit(0);
      return prob;
    }
    fseek(fptr, 0L, SEEK_END);
    long file_size = ftell(fptr);
    // each record is (int u, int v, float rate) = 12 bytes
    if(file_size <= 0 || file_size % 12 != 0)
    {
      fprintf(stderr,
          "file %s has invalid size %ld (must be a positive multiple of 12)\n",
          path.c_str(), file_size);
      fclose(fptr);
      exit(1);
    }
    prob.nnz = file_size/12;
    printf("prob.nnz = %lld\n", prob.nnz);

    mf_node *R = (mf_node *)sycl::malloc_host(sizeof(mf_node) * prob.nnz, q);

    rewind(fptr);

    long long idx = 0;
    while(true)
    {
      int flag = 0;
      int u,v;
      float r;

      flag += fread(&u, sizeof(int), 1, fptr); 
      flag += fread(&v, sizeof(int), 1, fptr); 
      flag += fread(&r, sizeof(float), 1, fptr); 
      if(flag != 3)break;

      // reject invalid node ids as early as possible; negative ids would
      // later map to an out-of-bounds grid index.
      if(u < 0 || v < 0)
      {
        fprintf(stderr, "invalid record %lld in %s: u=%d, v=%d\n",
            idx, path.c_str(), u, v);
        exit(1);
      }

      if(u + 1 > prob.m)prob.m = u + 1;
      if(v + 1 > prob.n)prob.n = v + 1;

      R[idx].u = u;
      R[idx].v = v;
      R[idx].rate = r;
      idx ++;
      //if(idx > 0 && idx%100000000 == 0)printf("progress: %%%.3f\n",100.0*idx/prob.nnz);
    }
    prob.R = R;

    fclose(fptr);

    printf("m:%d, n:%d, nnz:%lld\n",prob.m, prob.n, prob.nnz);
  }
  else
  {
    //data
    long long size_list[128];
    long long offset_list[128];
    pthread_t threads[128];
    pthread_arg pthread_arg_list[128];

    //get nnz & size_list
    FILE*fptrs[80];
    prob.nnz = 0;
    for(int i = 0;i < num_files;i++)
    {
      fptrs[i] = fopen(file_names[i].c_str(), "rb");
      fseek(fptrs[i], 0L, SEEK_END);
      size_list[i] = ftell(fptrs[i])/12;
      prob.nnz +=  size_list[i];
      fclose(fptrs[i]);
    }

    //get offset_list
    offset_list[0] = 0;
    for(int i = 1;i < num_files;i++)
    {
      offset_list[i] = offset_list[i-1] + size_list[i-1];
    }

    //malloc
    mf_node *R = (mf_node *)sycl::malloc_host(sizeof(mf_node) * prob.nnz, q);
    prob.R = R;

    //launch
    for(int i = 0;i < num_files; i++)
    {
      pthread_arg_list[i].thread_id = i;
      pthread_arg_list[i].path = file_names[i];
      pthread_arg_list[i].R = prob.R;
      pthread_arg_list[i].offset = offset_list[i];
      pthread_arg_list[i].size = size_list[i];
      pthread_create(&(threads[i]), NULL, read_problem_thread, (void*)(&(pthread_arg_list[i])));
    }

    for(int i = 0;i < num_files;i++)
    {
      pthread_join(threads[i], NULL);
    }
    prob.m = -1;
    prob.n = -1;
    for(int i = 0;i < num_files;i++)
    {
      if(pthread_arg_list[i].max_m >= prob.m) prob.m = pthread_arg_list[i].max_m;
      if(pthread_arg_list[i].max_n >= prob.n) prob.n = pthread_arg_list[i].max_n;
    }
    printf("m:%d, n:%d, nnz:%lld\n",prob.m, prob.n, prob.nnz);
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed = end.tv_sec - begin.tv_sec;
  elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
  printf("time elapsed:%.8fs\n\n\n",elapsed);

  return prob;
}

void grid_problem(mf_problem* prob)
{
  printf("grid problem ...\n");

  struct timespec begin, end;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &begin);

  //grid the problem into several grids
  long long u_seg, v_seg;
  if(prob->ux == 1)u_seg = prob->m;
  else u_seg = (long long)ceil((double)prob->m/prob->ux);
  if(prob->vy == 1)v_seg = prob->n;
  else v_seg = (long long)ceil((double)prob->n/prob->vy);

  prob->u_seg = u_seg;
  prob->v_seg = v_seg;

  auto get_grid_id = [=](int u, int v)
  {
    return ((u/u_seg)*prob->vy + v/v_seg);
  };

  //count the size of each grid
  prob->gridSize = new long long[prob->ux*prob->vy]();

  long long *gridSize = prob->gridSize;
  for(long long i = 0;i < prob->nnz;i++)
  {
    int tmp_u = prob->R[i].u;
    int tmp_v = prob->R[i].v;
    // guard against out-of-bounds indexing: negative or out-of-range
    // node ids would map to an invalid grid id and corrupt memory.
    if(tmp_u < 0 || tmp_u >= prob->m || tmp_v < 0 || tmp_v >= prob->n)
    {
      fprintf(stderr,
          "invalid record %lld: u=%d (m=%d), v=%d (n=%d)\n",
          i, tmp_u, prob->m, tmp_v, prob->n);
      exit(1);
    }
    gridSize[get_grid_id(tmp_u, tmp_v)] ++;
  }

  long long max_grid_size = 0;
  for(int i = 0;i < prob->ux*prob->vy; i++)
  {
    //printf("gridSize[%d]:%lld\n",i,prob->gridSize[i]);
    if(max_grid_size < prob->gridSize[i])max_grid_size = prob->gridSize[i];
  }
  prob->maxGridSize = max_grid_size;

  //generate the pointer to each grid.
  mf_node**R2D = new mf_node*[prob->ux*prob->vy + 1];
  mf_node* R = prob->R;
  R2D[0] = R;
  for(int grid = 0;grid < prob->ux*prob->vy; grid++)R2D[grid + 1] = R2D[grid] + gridSize[grid];

  prob->R2D = R2D;

  //swap
  mf_node**pivots = new mf_node*[prob->ux*prob->vy];
  for(int i = 0;i < prob->ux*prob->vy; i++)pivots[i] = R2D[i];

  for(int grid = 0; grid < prob->ux*prob->vy; grid++)
  {
    for(mf_node*pivot = pivots[grid]; pivot != R2D[grid + 1];)
    {
      int corre_grid = get_grid_id(pivot->u, pivot->v);
      if(corre_grid == grid)
      {  
        pivot ++;
        continue;
      }
      mf_node *next = pivots[corre_grid];
      swap(*pivot, *next);
      pivots[corre_grid] ++;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed = end.tv_sec - begin.tv_sec;
  elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
  printf("time elapsed:%.8fs\n\n\n",elapsed);
}

float LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}

void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}

void init_rand_state(unsigned int seed, unsigned int *state,
                     sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  state[i] = seed ^ i;
  LCG_random_init(state+i);
}

void random_init(unsigned int *__restrict state,
                 int state_size, sycl::half *__restrict array,
                 long long array_size, long long k, float scale,
                 sycl::nd_item<1> &item)
{
  int tid = item.get_global_id(0);
  int state_id = tid % state_size;
  for (int i = 0; i < array_size;
       i += item.get_group_range(0) * item.get_local_range(0))
  {
    int idx = i + tid;
    if(idx >= array_size) break;
    array[idx] = sycl::vec<float, 1>{LCG_random(state + state_id) * scale}
                     .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  }
}

void init_feature(sycl::queue &q, short *feature_vec, int grid, long long seg, int k)
{
  float scale = (float)sqrt(1.0/k);

  sycl::half *gpu_vec = (sycl::half *)sycl::malloc_device(seg * k * sizeof(sycl::half), q);

  int state_size = (seg/256 + 1)*256;
  printf("state_size (a multiple of 256):%d\n", state_size);
  unsigned int* d_state = sycl::malloc_device<unsigned int>(state_size, q);

  sycl::range<1> gws_rng (state_size);
  sycl::range<1> lws_rng (256);
  q.parallel_for(sycl::nd_range<1>(gws_rng, lws_rng), [=](sycl::nd_item<1> item) {
    init_rand_state(5551212, d_state, item);
  });

  const int blockSize = 256;
  const int blockNum = (seg*k + 255)/256;
  printf("\tnumber of thread blocks:%d\n", blockNum);
  printf("\tarraysize:%lld\n", seg*k);

  for(int i = 0;i < grid; i++)
  {
    printf("grid:%d\n",i);
    sycl::range<1> gws_init (blockSize * blockNum);
    sycl::range<1> lws_init (blockSize);
    q.parallel_for(sycl::nd_range<1>(gws_init, lws_init), [=](sycl::nd_item<1> item) {
      random_init(d_state, state_size, gpu_vec, seg * k, k, scale, item);
    }).wait();
    q.memcpy(feature_vec + i * seg * k, gpu_vec, sizeof(sycl::half) * seg * k).wait();
  }

  sycl::free(d_state, q);
  sycl::free(gpu_vec, q);
}

mf_model* init_model(sycl::queue &q, mf_problem*prob, int k, float ave)
{
  printf("init model ...\n");
  struct timespec begin, end;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &begin);

  mf_model *model = new mf_model;
  float scale_factor = sqrtf(1.f/k);
  model->fun = 0;
  model->m = prob->m;
  model->n = prob->n;

  model->u_grid = prob->u_grid;
  model->v_grid = prob->v_grid;

  model->x_grid = prob->x_grid;
  model->y_grid = prob->y_grid;

  model->ux = prob->ux;
  model->vy = prob->vy;

  model->u_seg = prob->u_seg;
  model->v_seg = prob->v_seg;
  model->k = k;
  model->b = ave;

  //allocate memory
  model->floatp = (float *)sycl::malloc_host(
      sizeof(float) * model->ux * model->u_seg * k, q);
  model->floatq = (float *)sycl::malloc_host(
      sizeof(float) * model->vy * model->v_seg * k, q);

  model->halfp = (short *)sycl::malloc_host(
      sizeof(short) * model->ux * model->u_seg * k, q);
  model->halfq = (short *)sycl::malloc_host(
      sizeof(short) * model->vy * model->v_seg * k, q);

  //random init
  init_feature(q, model->halfp, model->ux, model->u_seg, k);
  init_feature(q, model->halfq, model->vy, model->v_seg, k);

  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed = end.tv_sec - begin.tv_sec;
  elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
  printf("time elapsed:%.8fs\n\n\n",elapsed);

  return model;
}

#include "sgd_k128_kernel_hogwild_warp32.h"
#include <cmath>

void init_rand_state(unsigned int seed, unsigned int *state, int size,
                     sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  state[i] = seed ^ i;
  if(i < size) LCG_random_init(state+i);
}

void transform_half(const sycl::half *__restrict gpu_half_feature,
                    float *__restrict gpu_float_feature,
                    long long vec_size, sycl::nd_item<1> &item)
{
  int tid = item.get_global_id(0);
  int number_threads = item.get_group_range(0) * item.get_local_range(0);

  for(long long i = tid;i < vec_size;i += number_threads)
  {
    gpu_float_feature[i] =
        sycl::vec<sycl::half, 1>{gpu_half_feature[i]}
            .convert<float, sycl::rounding_mode::automatic>()[0];
  }
}

void transform_feature_vector(sycl::queue &q, short *half_feature, float *float_feature,
                              int m, int grid, long long seg, int k)
{
  sycl::half *gpu_half_feature = 
    (sycl::half *)sycl::malloc_device(sizeof(sycl::half) * seg * k, q);

  float *gpu_float_feature = (float *)sycl::malloc_device(sizeof(float) * seg * k, q);

  for(int i = 0;i < grid;i++)
  {
    q.memcpy(gpu_half_feature, half_feature + i * seg * k,
                sizeof(sycl::half) * seg * k).wait();

    int num_blocks = (seg*k+255)/256;
    if(num_blocks > 8*24)num_blocks = 8*24;

    sycl::range<1> gws (num_blocks * 256);
    sycl::range<1> lws (256);
    q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      transform_half(gpu_half_feature, gpu_float_feature, seg * k, item);
    }).wait();

    q.memcpy(float_feature + i * seg * k, gpu_float_feature,
             sizeof(float) * seg * k).wait();
  }

  sycl::free(gpu_half_feature, q);
  sycl::free(gpu_float_feature, q);
}

void sgd_update_k128(sycl::queue &q, Parameter para, mf_model *model, mf_problem *prob, float scale)
{
  printf("sgd_update_k128 ...\n");

  struct timespec begin, end;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &begin);

  //generate the random state for the hogwild scheduling policy.
  unsigned int *rand_state = sycl::malloc_device<unsigned int>(para.num_workers, q);

  sycl::range<1> gws_rng ((para.num_workers + 255) / 256 * 256);
  sycl::range<1> lws_rng (256);
  q.parallel_for(sycl::nd_range<1>(gws_rng, lws_rng), [=](sycl::nd_item<1> item) {
    init_rand_state(5551212, rand_state, para.num_workers, item);
  });

  //generate the dynamic learning rate
  float dynamic_rate[1024];
  float alpha = para.alpha;
  float beta  = para.beta;
  float lrate = para.lrate;

  for(int i = 0;i < (para.num_iters + 4);i++)
  {
    double tmp_rate = alpha / (1 + beta * pow(i, 1.5)) + lrate;
    dynamic_rate[i] = tmp_rate;
  }
  float *gpu_dynamic_rate = sycl::malloc_device<float>(1024, q);
  q.memcpy(gpu_dynamic_rate, dynamic_rate, sizeof(float) * 1024).wait();

  //malloc a problem grid on GPU
  if(prob->x_grid*prob->y_grid == 1)
  {
    prob->gpuR = (struct mf_node *)sycl::malloc_device(
        sizeof(mf_node) * prob->maxGridSize, q);
    prob->cur_u_id = -1;
    prob->cur_v_id = -1;
  }
  else
  {
    prob->gpuRptrs[0] = (struct mf_node *)sycl::malloc_device(
        sizeof(mf_node) * prob->maxGridSize, q);
    prob->gpuRptrs[1] = (struct mf_node *)sycl::malloc_device(
        sizeof(mf_node) * prob->maxGridSize, q);
    prob->cur_global_x_id[0] = -1;
    prob->cur_global_x_id[1] = -1;
    prob->cur_global_y_id[0] = -1;
    prob->cur_global_y_id[1] = -1;
  }

  //malloc feature vectors on GPU
  if(prob->x_grid*prob->y_grid == 1)
  {
    model->gpuHalfp = (sycl::half *)sycl::malloc_device(
        sizeof(sycl::half) * model->u_seg * model->k, q);
    model->gpuHalfq = (sycl::half *)sycl::malloc_device(
        sizeof(sycl::half) * model->v_seg * model->k, q);
    model->cur_u_id = -1;
    model->cur_v_id = -1;
  }
  else
  {
    model->gpuHalfPptrs[0] = (sycl::half *)sycl::malloc_device(
        sizeof(sycl::half) * model->u_seg * model->k, q);
    model->gpuHalfPptrs[1] = (sycl::half *)sycl::malloc_device(
        sizeof(sycl::half) * model->u_seg * model->k, q);
    model->gpuHalfQptrs[0] = (sycl::half *)sycl::malloc_device(
        sizeof(sycl::half) * model->v_seg * model->k, q);
    model->gpuHalfQptrs[1] = (sycl::half *)sycl::malloc_device(
        sizeof(sycl::half) * model->v_seg * model->k, q);

    model->cur_global_x_id[0] = -1;
    model->cur_global_x_id[1] = -1;
    model->cur_global_y_id[0] = -1;
    model->cur_global_y_id[1] = -1;
  }   

  //set update count
  int update_vector_size = 128;
  int *update_count_per_block = new int[prob->ux*prob->vy]();
  int max_update_count_per_block = -1;
  for(int cur_grid_id = 0;cur_grid_id < prob->ux*prob->vy; cur_grid_id ++)
  {
    update_count_per_block[cur_grid_id] = 
      (ceil)(1.0*prob->gridSize[cur_grid_id]/(para.num_workers*update_vector_size));   
    if(max_update_count_per_block < update_count_per_block[cur_grid_id])
    {
      max_update_count_per_block = update_count_per_block[cur_grid_id];
    }
  }

  // random shuffle
  random_device rd;
  mt19937 g(rd());

  //run the update kernel
  if(prob->u_grid*prob->v_grid == 1)
  {
    q.memcpy(prob->gpuR, prob->R2D[0], sizeof(mf_node) * prob->gridSize[0]);
    q.memcpy(model->gpuHalfp, model->halfp, sizeof(sycl::half) * model->u_seg * model->k);
    q.memcpy(model->gpuHalfq, model->halfq, sizeof(sycl::half) * model->v_seg * model->k);
    q.wait();

    sycl::range<1> gws (para.num_workers / 4 * 128);
    sycl::range<1> lws (128);
    q.submit([&](sycl::handler &cgh) {
      auto prob_gpuR = prob->gpuR;
      auto prob_gridSize = prob->gridSize[0];
      auto model_gpuHalfp = model->gpuHalfp;
      auto model_gpuHalfq = model->gpuHalfq;
      auto model_u_seg = model->u_seg;
      auto model_v_seg = model->v_seg;
      auto model_k = model->k;
      auto update_count_per_block0 = update_count_per_block[0];
      auto prob_u_grid = prob->u_grid;
      auto prob_v_grid = prob->v_grid;
      auto lambda_p = para.lambda_p; 
      auto lambda_q = para.lambda_q; 

      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(32)]] {
            sgd_k128_kernel_hogwild_warp32_lrate(
                prob_gpuR, prob_gridSize, model_gpuHalfp,
                model_gpuHalfq, rand_state, gpu_dynamic_rate,
                model_u_seg, model_v_seg, model_k, para.num_iters,
                0, max_update_count_per_block, update_count_per_block0,
                update_vector_size, lambda_p, lambda_q,
                prob_u_grid, prob_v_grid, 0, 0, item);
          });
    }).wait();

    q.memcpy(model->halfp, model->gpuHalfp, sizeof(sycl::half) * model->u_seg * model->k);
    q.memcpy(model->halfq, model->gpuHalfq, sizeof(sycl::half) * model->v_seg * model->k);
    q.wait();
  }
  else if(prob->x_grid*prob->y_grid == 1)
  {
    //random shuffle
    vector<int> u_id_vec(prob->u_grid, 0);
    vector<int> v_id_vec(prob->v_grid, 0);
    for(int i = 0;i < prob->u_grid;i++) u_id_vec[i] = i;
    for(int i = 0;i < prob->v_grid;i++) v_id_vec[i] = i;

    for(int iter = 0;iter < para.num_iters; iter ++)
    {
      shuffle(u_id_vec.begin(), u_id_vec.end(), g);
      for(int u_ite = 0;u_ite < prob->u_grid; u_ite ++)
      {

        shuffle(v_id_vec.begin(), v_id_vec.end(), g);
        for(int v_ite = 0;v_ite < prob->v_grid; v_ite ++)
        {
          int cur_u_id = u_id_vec[u_ite];
          int cur_v_id = v_id_vec[v_ite];

          int cur_grid_id = cur_u_id*prob->v_grid + cur_v_id;
          //transfer problem grid to gpu.
          if(prob->cur_u_id != cur_u_id || prob->cur_v_id != cur_v_id)
          {
            q.memcpy(prob->gpuR, prob->R2D[cur_grid_id],
                     sizeof(mf_node) * prob->gridSize[cur_grid_id]).wait();
          }

          prob->cur_u_id = cur_u_id;
          prob->cur_v_id = cur_v_id;

          //transfer p grid to gpu
          if(model->cur_u_id == -1)
          {
            short *p_tmp = model->halfp + model->u_seg*model->k*cur_u_id;
            q.memcpy(model->gpuHalfp, p_tmp,
                     sizeof(sycl::half) * model->u_seg * model->k).wait();
          }
          else if(model->cur_u_id != cur_u_id)
          {
            short *p_tmp = model->halfp + model->u_seg*model->k*model->cur_u_id;
            q.memcpy(p_tmp, model->gpuHalfp,
                     sizeof(sycl::half) * model->u_seg * model->k).wait();

            p_tmp = model->halfp + model->u_seg*model->k*cur_u_id;
            q.memcpy(model->gpuHalfp, p_tmp,
                     sizeof(sycl::half) * model->u_seg * model->k).wait();
          }
          model->cur_u_id = cur_u_id;

          //transfer q grid to gpu
          if(model->cur_v_id == -1)
          {
            short *q_tmp = model->halfq + model->v_seg*model->k*cur_v_id;
            q.memcpy(model->gpuHalfq, q_tmp,
                     sizeof(sycl::half) * model->v_seg * model->k).wait();
          }
          else if(model->cur_v_id != cur_v_id)
          {
            short *q_tmp = model->halfq + model->v_seg*model->k*model->cur_v_id;
            q.memcpy(q_tmp, model->gpuHalfq,
                     sizeof(sycl::half) * model->v_seg * model->k).wait();

            q_tmp = model->halfq + model->v_seg*model->k*cur_v_id;
            q.memcpy(model->gpuHalfq, q_tmp,
                     sizeof(sycl::half) * model->v_seg * model->k).wait();
          }
          model->cur_v_id = cur_v_id;

          //call the kernel
          sycl::range<1> gws (para.num_workers / 4 * 128);
          sycl::range<1> lws (128);
          q.submit([&](sycl::handler &cgh) {
            auto prob_gpuR = prob->gpuR;
            auto prob_gridSize_cur_grid_id = prob->gridSize[cur_grid_id];
            auto model_gpuHalfp = model->gpuHalfp;
            auto model_gpuHalfq = model->gpuHalfq;
            auto model_u_seg = model->u_seg;
            auto model_v_seg = model->v_seg;
            auto model_k = model->k;
            auto update_count_per_block_cur_grid_id =
                update_count_per_block[cur_grid_id];
            auto prob_u_grid = prob->u_grid;
            auto prob_v_grid = prob->v_grid;
            auto lambda_p = para.lambda_p;
            auto lambda_q = para.lambda_q;

            cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item)
              [[intel::reqd_sub_group_size(32)]] {
              sgd_k128_kernel_hogwild_warp32_lrate(
                prob_gpuR, prob_gridSize_cur_grid_id,
                model_gpuHalfp, model_gpuHalfq, rand_state,
                gpu_dynamic_rate, model_u_seg, model_v_seg,
                model_k, 1, iter, max_update_count_per_block,
                update_count_per_block_cur_grid_id,
                update_vector_size, lambda_p, lambda_q,
                prob_u_grid, prob_v_grid, cur_u_id,
                cur_v_id, item);
            });
          }).wait();
        }
      }
      q.wait();
    }
    q.wait();

    //printf("%d,%d\n", model->cur_u_id, model->cur_v_id);

    //transfer p back to CPU
    if(model->cur_u_id >= 0)
    {
      short *p_tmp = model->halfp + model->u_seg*model->k*model->cur_u_id;
      q.memcpy(p_tmp, model->gpuHalfp, sizeof(sycl::half) * model->u_seg * model->k).wait();
    }
    //transfer q back to CPU
    if(model->cur_v_id >= 0)
    {
      short *q_tmp = model->halfq + model->v_seg*model->k*model->cur_v_id;
      q.memcpy(q_tmp, model->gpuHalfq, sizeof(sycl::half) * model->v_seg * model->k).wait();
    }
  }
  else
  {
    //scheduling info
    int *global_x_list = new int[prob->x_grid*prob->y_grid];
    int *global_y_list = new int[prob->x_grid*prob->y_grid];
    int *global_id_list = new int[prob->x_grid*prob->y_grid];

    //random shuffle
    vector<int> u_id_vec(prob->u_grid, 0);
    vector<int> v_id_vec(prob->v_grid, 0);
    for(int i = 0;i < prob->u_grid;i++)u_id_vec[i] = i;
    for(int i = 0;i < prob->v_grid;i++)v_id_vec[i] = i;

    vector<int> x_id_vec(prob->x_grid, 0);
    vector<int> y_id_vec(prob->y_grid, 0);
    for(int i = 0;i < prob->x_grid;i++)x_id_vec[i] = i;
    for(int i = 0;i < prob->y_grid;i++)y_id_vec[i] = i;

    //fully random
    vector<int> uv_id_vec(prob->u_grid*prob->v_grid, 0);
    for(int i = 0;i < prob->u_grid*prob->v_grid; i++)uv_id_vec[i] = i;
    vector<int> xy_id_vec(prob->x_grid*prob->y_grid, 0);
    for(int i = 0;i < prob->x_grid*prob->y_grid; i++)xy_id_vec[i] = i;

    for(int iter = 0;iter < para.num_iters; iter ++)
    {
      shuffle(uv_id_vec.begin(), uv_id_vec.end(), g);
      shuffle(u_id_vec.begin(), u_id_vec.end(), g);

      for(int u_ite = 0;u_ite < prob->u_grid; u_ite ++)
      {
        shuffle(v_id_vec.begin(), v_id_vec.begin(), g);
        for(int v_ite = 0;v_ite < prob->v_grid; v_ite ++)
        {

          //fully random
          int tmp_uv_id = u_ite*prob->v_grid + v_ite;
          int cur_u_id = uv_id_vec[tmp_uv_id]/prob->v_grid;
          int cur_v_id = uv_id_vec[tmp_uv_id]%prob->v_grid;

          //set information
          shuffle(x_id_vec.begin(), x_id_vec.end(), g);
          shuffle(xy_id_vec.begin(), xy_id_vec.end(), g);

          for(int local_x_ite = 0;local_x_ite < prob->x_grid;local_x_ite ++)
          {
            shuffle(y_id_vec.begin(),y_id_vec.end(), g);
            for(int local_y_ite = 0;local_y_ite < prob->y_grid;local_y_ite ++)
            {

              //fully random
              int tmp_xy_id = local_x_ite*prob->y_grid + local_y_ite;
              int cur_x_id = xy_id_vec[tmp_xy_id]/prob->y_grid;
              int cur_y_id = xy_id_vec[tmp_xy_id]%prob->y_grid;

              int local_id = cur_x_id*prob->y_grid + cur_y_id;

              int global_x = cur_u_id*prob->x_grid + cur_x_id;
              int global_y = cur_v_id*prob->y_grid + cur_y_id;
              int global_id = global_x*prob->vy + global_y;

              global_x_list[local_id] = global_x;
              global_y_list[local_id] = global_y;
              global_id_list[local_id] = global_id;

            }
          }

          //run
          for(int i = -1;i < prob->x_grid*prob->y_grid;i++)
          {
            //compute
            if(i >= 0)
            {
              sycl::range<1> gws (para.num_workers / 4 * 128);
              sycl::range<1> lws (128);
              q.submit([&](sycl::handler &cgh) {
                auto prob_gpuRptrs_i_ct0 = prob->gpuRptrs[i % 2];
                auto prob_gridSize_global_id_list_i_ct1 =
                    prob->gridSize[global_id_list[i]];
                auto model_gpuHalfPptrs_i_ct2 = model->gpuHalfPptrs[i % 2];
                auto model_gpuHalfQptrs_i_ct3 = model->gpuHalfQptrs[i % 2];
                auto model_u_seg_ct6 = model->u_seg;
                auto model_v_seg_ct7 = model->v_seg;
                auto model_k_ct8 = model->k;
                auto update_count_per_block_global_id_list_i_ct12 =
                    update_count_per_block[global_id_list[i]];
                auto prob_ux_ct16 = prob->ux;
                auto prob_vy_ct17 = prob->vy;
                auto global_x_list_i_ct18 = global_x_list[i];
                auto global_y_list_i_ct19 = global_y_list[i];
                auto lambda_p = para.lambda_p; 
                auto lambda_q = para.lambda_q; 

                cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item)
                  [[intel::reqd_sub_group_size(32)]] {
                  sgd_k128_kernel_hogwild_warp32_lrate(
                    prob_gpuRptrs_i_ct0,
                    prob_gridSize_global_id_list_i_ct1,
                    model_gpuHalfPptrs_i_ct2,
                    model_gpuHalfQptrs_i_ct3, rand_state,
                    gpu_dynamic_rate, model_u_seg_ct6,
                    model_v_seg_ct7, model_k_ct8, 1, iter,
                    max_update_count_per_block,
                    update_count_per_block_global_id_list_i_ct12,
                    update_vector_size, lambda_p, lambda_q,
                    prob_ux_ct16, prob_vy_ct17, global_x_list_i_ct18,
                    global_y_list_i_ct19, item);
                });
              }).wait();
            }

            //memcpy for the next block
            if(i != (prob->x_grid*prob->y_grid - 1))
            {
              int next_global_x = global_x_list[i+1];
              int next_global_y = global_y_list[i+1];
              int next_global_id = global_id_list[i+1];

              //transfer problem grid to gpu
              if(prob->cur_global_x_id[(i+1)%2] !=  next_global_x || prob->cur_global_y_id[(i+1)%2] != next_global_y)
              {
                q.memcpy(prob->gpuRptrs[(i + 1) % 2], prob->R2D[next_global_id],
                    sizeof(mf_node) * prob->gridSize[next_global_id]);
              }

              //transfer feature p
              if(model->cur_global_x_id[(i+1)%2] == -1)
              {
                if(model->cur_global_x_id[(i+2)%2] == next_global_x)
                {
                  model->cur_global_x_id[(i+2)%2] = -1;
                  model->cur_global_x_id[(i+1)%2] = next_global_x;

                  sycl::half *tmp_ptr = model->gpuHalfPptrs[(i + 1) % 2];
                  model->gpuHalfPptrs[(i+1)%2] = model->gpuHalfPptrs[(i+2)%2];
                  model->gpuHalfPptrs[(i+2)%2] = tmp_ptr;
                }
                else
                {
                  short *p_tmp = model->halfp + model->u_seg*model->k*next_global_x;
                  q.memcpy(model->gpuHalfPptrs[(i + 1) % 2], p_tmp,
                      sizeof(sycl::half) * model->u_seg * model->k);
                  model->cur_global_x_id[(i+1)%2] = next_global_x;
                }
              }
              else if(model->cur_global_x_id[(i+1)%2] != next_global_x)
              {
                if(model->cur_global_x_id[(i+2)%2] == -1)
                {
                  //swap value
                  int tmp = model->cur_global_x_id[(i+1)%2];
                  model->cur_global_x_id[(i+1)%2] = next_global_x;
                  model->cur_global_x_id[(i+2)%2] = tmp;

                  //swap pointer
                  sycl::half *p_tmp = model->gpuHalfPptrs[(i + 1) % 2];
                  model->gpuHalfPptrs[(i+1)%2] = model->gpuHalfPptrs[(i+2)%2];
                  model->gpuHalfPptrs[(i+2)%2] = p_tmp;

                  //transfer
                  short *p_tmp_trans = model->halfp + model->u_seg*model->k*next_global_x;
                  q.memcpy(
                      model->gpuHalfPptrs[(i + 1) % 2], p_tmp_trans,
                      sizeof(sycl::half) * model->u_seg * model->k);
                  model->cur_global_x_id[(i+1)%2] = next_global_x;
                }
                else if(model->cur_global_x_id[(i+2)%2] == next_global_x)
                {
                  //swap value
                  int tmp = model->cur_global_x_id[(i+1)%2];
                  model->cur_global_x_id[(i+1)%2] = next_global_x;
                  model->cur_global_x_id[(i+2)%2] = tmp;

                  //swap pointer
                  sycl::half *p_tmp = model->gpuHalfPptrs[(i + 1) % 2];
                  model->gpuHalfPptrs[(i+1)%2] = model->gpuHalfPptrs[(i+2)%2];
                  model->gpuHalfPptrs[(i+2)%2] = p_tmp;
                }
                else
                {
                  short *p_tmp = model->halfp + model->u_seg*model->k*model->cur_global_x_id[(i+1)%2];
                  q.memcpy(
                      p_tmp, model->gpuHalfPptrs[(i + 1) % 2],
                      sizeof(sycl::half) * model->u_seg * model->k);

                  p_tmp = model->halfp + model->u_seg*model->k*next_global_x;
                  q.memcpy(
                      model->gpuHalfPptrs[(i + 1) % 2], p_tmp,
                      sizeof(sycl::half) * model->u_seg * model->k);

                  model->cur_global_x_id[(i+1)%2] = next_global_x;
                }
              }

              //transfer feature q
              if(model->cur_global_y_id[(i+1)%2] == -1)
              {
                if(model->cur_global_y_id[(i+2)%2] == next_global_y)
                {
                  model->cur_global_y_id[(i+2)%2] = -1;
                  model->cur_global_y_id[(i+1)%2] = next_global_y;

                  sycl::half *tmp_ptr = model->gpuHalfQptrs[(i + 1) % 2];
                  model->gpuHalfQptrs[(i+1)%2] = model->gpuHalfQptrs[(i+2)%2];
                  model->gpuHalfQptrs[(i+2)%2] = tmp_ptr;
                }
                else
                {
                  short *q_tmp = model->halfq + model->v_seg*model->k*next_global_y;
                  q.memcpy(
                      model->gpuHalfQptrs[(i + 1) % 2], q_tmp,
                      sizeof(sycl::half) * model->v_seg * model->k);
                  model->cur_global_y_id[(i+1)%2] = next_global_y;
                }
              }
              else if(model->cur_global_y_id[(i+1)%2] != next_global_y)
              {
                if(model->cur_global_y_id[(i+2)%2] == -1)
                {
                  //swap value
                  int tmp = model->cur_global_y_id[(i+1)%2];
                  model->cur_global_y_id[(i+1)%2] = model->cur_global_y_id[(i+2)%2];
                  model->cur_global_y_id[(i+2)%2] = tmp;

                  //swap pointer
                  sycl::half *q_tmp = model->gpuHalfQptrs[(i + 1) % 2];
                  model->gpuHalfQptrs[(i+1)%2] = model->gpuHalfQptrs[(i+2)%2];
                  model->gpuHalfQptrs[(i+2)%2] = q_tmp;

                  short *q_tmp_trans = model->halfq + model->v_seg*model->k*next_global_y;
                  q.memcpy(
                      model->gpuHalfQptrs[(i + 1) % 2], q_tmp_trans,
                      sizeof(sycl::half) * model->v_seg * model->k);
                  model->cur_global_y_id[(i+1)%2] = next_global_y;
                }
                else if(model->cur_global_y_id[(i+2)%2] == next_global_y)
                {
                  //swap value
                  int tmp = model->cur_global_y_id[(i+1)%2];
                  model->cur_global_y_id[(i+1)%2] = model->cur_global_y_id[(i+2)%2];
                  model->cur_global_y_id[(i+2)%2] = tmp;

                  //swap pointer
                  sycl::half *q_tmp = model->gpuHalfQptrs[(i + 1) % 2];
                  model->gpuHalfQptrs[(i+1)%2] = model->gpuHalfQptrs[(i+2)%2];
                  model->gpuHalfQptrs[(i+2)%2] = q_tmp;
                }
                else
                {
                  short *q_tmp = model->halfq + model->v_seg*model->k*model->cur_global_y_id[(i+1)%2];
                  q.memcpy(q_tmp, model->gpuHalfQptrs[(i + 1) % 2],
                      sizeof(sycl::half) * model->v_seg * model->k);

                  q_tmp = model->halfq + model->v_seg*model->k*next_global_y;
                  q.memcpy(
                      model->gpuHalfQptrs[(i + 1) % 2], q_tmp,
                      sizeof(sycl::half) * model->v_seg * model->k);
                  model->cur_global_y_id[(i+1)%2] = next_global_y;
                }
              }
            }
            q.wait();
          }
        }
      }
      q.wait();
    }
    q.wait();

    //transfer p back
    if(model->cur_global_x_id[0] != -1)
    {
      short *p_tmp = model->halfp + model->u_seg*model->k*model->cur_global_x_id[0];
      q.memcpy(p_tmp, model->gpuHalfPptrs[0], sizeof(sycl::half) * model->u_seg * model->k).wait();
    }
    if(model->cur_global_x_id[1] != -1)
    {
      short *p_tmp = model->halfp + model->u_seg*model->k*model->cur_global_x_id[1];
      q.memcpy(p_tmp, model->gpuHalfPptrs[1], sizeof(sycl::half) * model->u_seg * model->k).wait();
    }

    //transfer q back
    if(model->cur_global_y_id[0] != -1)
    {
      short *q_tmp = model->halfq + model->v_seg*model->k*model->cur_global_y_id[0];
      q.memcpy(q_tmp, model->gpuHalfQptrs[0], sizeof(sycl::half) * model->v_seg * model->k).wait();
    }
    if(model->cur_global_y_id[1] != -1)
    {
      short *q_tmp = model->halfq + model->v_seg*model->k*model->cur_global_y_id[1];
      q.memcpy(q_tmp, model->gpuHalfQptrs[1], sizeof(sycl::half) * model->v_seg * model->k).wait();
    }
  }

  if(prob->x_grid*prob->y_grid == 1)
  {
    sycl::free(model->gpuHalfp, q);
    sycl::free(model->gpuHalfq, q);
    sycl::free(prob->gpuR, q);
  }
  else
  {
    sycl::free(model->gpuHalfPptrs[0], q);
    sycl::free(model->gpuHalfPptrs[1], q);
    sycl::free(model->gpuHalfQptrs[0], q);
    sycl::free(model->gpuHalfQptrs[1], q);
    sycl::free(prob->gpuRptrs[0], q);
    sycl::free(prob->gpuRptrs[1], q);
  }

  //transform halfp & halfq to floatp & floatq.
  transform_feature_vector(q, model->halfp, model->floatp, model->m, model->ux, model->u_seg, model->k);
  transform_feature_vector(q, model->halfq, model->floatq, model->n, model->vy, model->v_seg, model->k);

  sycl::free(gpu_dynamic_rate, q);
  sycl::free(rand_state, q);

  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed = end.tv_sec - begin.tv_sec;
  elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
  printf("time elapsed:%.8fs\n\n\n",elapsed);
}

void scale_model(mf_model *model, float scale)
{
  printf("scale model ...\n");

  struct timespec begin, end;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &begin);

  float factor_scale = sqrt(scale);
  for(long long i = 0; i < ((long long)model->m)*model->k; i++)model->floatp[i] = model->floatp[i]*factor_scale;


  for(long long i = 0; i < model->n*model->k; i++)model->floatq[i] = model->floatq[i]*factor_scale;

  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed = end.tv_sec - begin.tv_sec;
  elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
  printf("time elapsed:%.8fs\n\n\n",elapsed);
}


void shuffle_model(mf_model *model, int* inv_p_map, int* inv_q_map)
{
  printf("shuffle model ...\n");

  struct timespec begin, end;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &begin);

  auto inv_shuffle1 = [] (float *vec, int *map, int size, int k)
  {
    for(int pivot = 0; pivot < size;)
    {
      if(pivot == map[pivot])
      {
        ++pivot;
        continue;
      }

      int next = map[pivot];

      for(SGDIndex d = 0; d < k; d++)swap(*(vec + (long long)pivot*k+d), *(vec+(long long)next*k+d));

      map[pivot] = map[next];
      map[next] = next;
    }
  };

  inv_shuffle1(model->floatp, inv_p_map, model->m, model->k);
  inv_shuffle1(model->floatq, inv_q_map, model->n, model->k);

  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed = end.tv_sec - begin.tv_sec;
  elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
  printf("time elapsed:%.8fs\n\n\n",elapsed);
}

//the core computation function
mf_model*sgd_train(sycl::queue &q, mf_problem*tr, mf_problem*te, Parameter para)
{
  printf("sgd_train called\n");

  //collect the factor. scaling is used to make sure every rating is around 1.
  SGDRate ave;
  SGDRate std_dev;
  SGDRate scale = 1.0;

  collect_data(tr, ave, std_dev);
  scale = fmaxf((SGDRate)1e-4, std_dev);

  //shuffle the u & v randomly to: 1) increase randomness. 2) block balance.
  int* p_map = gen_random_map(tr->m);
  int* q_map = gen_random_map(tr->n);
  int* inv_p_map = gen_inv_map(p_map, tr->m);
  int* inv_q_map = gen_inv_map(q_map, tr->n);

  shuffle_problem(tr, p_map, q_map);

  grid_problem(tr); 

  //scale problem
  scale_problem(tr, 1.0/scale, tr->u_seg, tr->v_seg);
  para.lambda_p = para.lambda_p/scale;
  para.lambda_q = para.lambda_q/scale;

  //init model
  mf_model*model = init_model(q, tr, para.k, ave/std_dev);

  //train
  sgd_update_k128(q, para, model, tr, scale);

  //scale model
  scale_model(model, scale);

  //shuffle model
  shuffle_model(model, inv_p_map, inv_q_map);

  return model;
}
