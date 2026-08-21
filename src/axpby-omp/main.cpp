#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include <omp.h>

// Multi-tensor AXPBY:  out = a * x + b * y  (elementwise, per tensor)
// The CUDA versions use a chunked "multi_tensor_apply" launcher: the work
// is split into fixed-size chunks and each thread block processes one chunk of
// one tensor, so a single kernel launch handles many (small) tensors at once.
// This OpenMP offload version mirrors that design: a flat list of
// (tensor, chunk) "blocks" is built on the host and processed in a single
// target region where each team handles one chunk.

template <typename T>
struct Tensor {
  T* data_ptr;
  int64_t numel;
};

template<typename scalar_t>
void multi_tensor_axpby(int chunk_size,
                        std::vector<std::vector<Tensor<scalar_t>>> &tensor_lists,
                        float a, float b) {
  const int max_tensors = tensor_lists[0].size();
  const int dev = omp_get_default_device();

  auto start = std::chrono::steady_clock::now();

  // Build the flat chunk metadata (mirrors the CUDA multi_tensor_apply
  // launcher, where each block maps to one chunk of one tensor). Store device
  // addresses so the single target region can index straight into device
  // memory without any per-tensor mapping.
  std::vector<scalar_t*> chunk_x, chunk_y, chunk_out;
  std::vector<int64_t> chunk_offset, chunk_len;
  for (int n = 0; n < max_tensors; n++) {
    scalar_t *x = (scalar_t*) omp_get_mapped_ptr(tensor_lists[0][n].data_ptr, dev);
    scalar_t *y = (scalar_t*) omp_get_mapped_ptr(tensor_lists[1][n].data_ptr, dev);
    scalar_t *out = (scalar_t*) omp_get_mapped_ptr(tensor_lists[2][n].data_ptr, dev);
    const int64_t len = tensor_lists[0][n].numel;
    const int64_t chunks_this_tensor = (len + chunk_size - 1) / chunk_size;
    for (int64_t c = 0; c < chunks_this_tensor; c++) {
      const int64_t offset = c * (int64_t)chunk_size;
      chunk_x.push_back(x);
      chunk_y.push_back(y);
      chunk_out.push_back(out);
      chunk_offset.push_back(offset);
      chunk_len.push_back(std::min<int64_t>(chunk_size, len - offset));
    }
  }

  const int nblocks = (int) chunk_x.size();
  scalar_t **xp = chunk_x.data();
  scalar_t **yp = chunk_y.data();
  scalar_t **op = chunk_out.data();
  int64_t *coff = chunk_offset.data();
  int64_t *clen = chunk_len.data();

  // One team per chunk ("block"); threads within a team stride over the chunk.
  #pragma omp target teams distribute \
      map(to: xp[0:nblocks], yp[0:nblocks], op[0:nblocks], \
              coff[0:nblocks], clen[0:nblocks])
  for (int blk = 0; blk < nblocks; blk++) {
    scalar_t *x = xp[blk];
    scalar_t *y = yp[blk];
    scalar_t *out = op[blk];
    const int64_t off = coff[blk];
    const int64_t n = clen[blk];
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
      out[off + i] = a * (float)x[off + i] + b * (float)y[off + i];
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Chunk size %8d | Total execution time of multi_tensor_axpby: %f (us)\n",
         chunk_size, (time * 1e-3f));
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <number of tensors>\n", argv[0]);
    return 1;
  }
  const int max_tensors = atoi(argv[1]);

  std::vector<std::vector<Tensor<float>>> tensor_lists_ref (3);
  std::vector<std::vector<Tensor<float>>> tensor_lists (3);
  srand(123);
  for (int n = 0; n < max_tensors; n++) {
    int64_t length = rand() % (1024*1024) + 1024;
    for (int d = 0; d < 3; d++) {
      float *tensor = (float*) malloc (sizeof(float) * length);
      float *tensor_ref = (float*) malloc (sizeof(float) * length);
      if (d <= 1) {
        for (int64_t i = 0; i < length; i++)
          tensor[i] = tensor_ref[i] = rand() % length;
      }
      #pragma omp target enter data map(to: tensor[0:length])

      Tensor<float> t;
      t.data_ptr = tensor;
      t.numel = length;
      tensor_lists[d].push_back(t);

      Tensor<float> t_ref;
      t_ref.data_ptr = tensor_ref;
      t_ref.numel = length;
      tensor_lists_ref[d].push_back(t_ref);
    }
  }

  const float a = 1.f;
  const float b = 1.f;

  for (int chunk_size = 256; chunk_size <= 1024*1024; chunk_size = chunk_size * 2) {

    multi_tensor_axpby<float>(chunk_size, tensor_lists, a, b);

    bool ok  = true;
    for (int n = 0; n < max_tensors; n++) {
      auto x = tensor_lists_ref[0][n];
      auto y = tensor_lists_ref[1][n];
      auto z = tensor_lists_ref[2][n];
      for (int i = 0; i < x.numel; i++) {
        z.data_ptr[i] = a * x.data_ptr[i] + b * y.data_ptr[i];
      }
      float *zp = tensor_lists[2][n].data_ptr;
      int64_t len = tensor_lists[2][n].numel;
      #pragma omp target update from(zp[0:len])
      for (int i = 0; i < x.numel; i++) {
        if (fabsf(zp[i] - z.data_ptr[i]) > 1e-3f) {
          ok = false;
          break;
        }
      }
      if (!ok) break;
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  for (int d = 0; d < 3; d++) {
    for (int t = 0; t < max_tensors; t++) {
       float *tensor = tensor_lists[d][t].data_ptr;
       int64_t len = tensor_lists[d][t].numel;
       #pragma omp target exit data map(delete: tensor[0:len])
       free(tensor_lists[d][t].data_ptr);
       free(tensor_lists_ref[d][t].data_ptr);
    }
  }

  return 0;
}
