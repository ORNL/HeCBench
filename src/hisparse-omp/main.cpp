// One HiSparse decode step, in program order:
//   1. backup        copy newly written device rows into the pinned host pool
//   2. invalidate    drop residency entries for those written global slots
//   3. resolve       assign hot slots for the sparse top-k
//   4. gather_compact  load misses into this layer's hot cache
//   5. gather_plan     replay that plan into the next layer's hot cache
//
// The step uses the kernels' full addressing: request ids, a permuted state
// table, source and resident block tables, a padding row, a fully resident
// row, and split value/scale cache lines. OpenMP target offload port: every
// target region and copy is synchronous on the default device.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <omp.h>

#include "hisparse_kernels.h"
#include "reference.h"

// Return 1 from the enclosing function when call yields a nonzero status.
// The callee has already printed the reason.
#define CHECK(call)                                                            \
  do {                                                                         \
    if ((call) != 0) return 1;                                                 \
  } while (0)

// Owns every device allocation and host pool on the default OpenMP device,
// frees them on destruction (including on the CHECK early returns), and
// moves bytes between host and device memory.
struct OmpArena {
  int device_id = omp_get_default_device();
  int host_id = omp_get_initial_device();
  std::vector<std::pair<char*, size_t>> device;
  std::vector<void*> pinned;

  // omp_target_free every recorded device pointer and release every
  // recorded host pool. Only successful allocations are recorded.
  ~OmpArena() {
    for (const auto& range : device) omp_target_free(range.first, device_id);
    for (void* p : pinned) free_host(p);
  }

  OmpArena() = default;
  OmpArena(const OmpArena&) = delete;
  OmpArena& operator=(const OmpArena&) = delete;

  // omp_target_alloc count elements of T. On success set out, record the
  // pointer and size, and return 0. On failure set out to nullptr, record
  // nothing, print the size, and return 1.
  template <typename T>
  int alloc_device(T*& out, size_t count) {
    const size_t bytes = count * sizeof(T);
    out = static_cast<T*>(omp_target_alloc(bytes, device_id));
    if (out == nullptr) {
      fprintf(stderr, "omp_target_alloc of %zu bytes failed\n", bytes);
      return 1;
    }
    device.emplace_back(reinterpret_cast<char*>(out), bytes);
    return 0;
  }

  // Allocate bytes bytes of host memory that target regions can read and
  // write through the same pointer: omp_target_alloc_host with the Intel
  // compiler, the llvm_omp_target_host_mem_alloc allocator otherwise. On
  // success set out, record the pointer, and return 0. On failure set out to
  // nullptr, record nothing, print the size, and return 1.
  int alloc_pinned(char*& out, size_t bytes) {
#if defined(__INTEL_LLVM_COMPILER)
    out = static_cast<char*>(omp_target_alloc_host(bytes, device_id));
#else
    out = static_cast<char*>(omp_alloc(bytes, llvm_omp_target_host_mem_alloc));
#endif
    if (out == nullptr) {
      fprintf(stderr, "host pool allocation of %zu bytes failed\n", bytes);
      return 1;
    }
    pinned.push_back(out);
    return 0;
  }

  // Release a pointer from alloc_pinned with the matching call.
  void free_host(void* p) {
#if defined(__INTEL_LLVM_COMPILER)
    omp_target_free(p, device_id);
#else
    omp_free(p, llvm_omp_target_host_mem_alloc);
#endif
  }

  // True when p lies inside a recorded device allocation.
  bool on_device(const void* p) const {
    const char* c = static_cast<const char*>(p);
    for (const auto& range : device) {
      if (c >= range.first && c < range.first + range.second) return true;
    }
    return false;
  }

  // Copy bytes from src to dst with omp_target_memcpy. Each side is the
  // device when it lies in a recorded device allocation and the host
  // otherwise (including the host pools). Return 0, or print and return 1
  // when the copy fails.
  int copy(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return 0;
    const int dst_id = on_device(dst) ? device_id : host_id;
    const int src_id = on_device(src) ? device_id : host_id;
    if (omp_target_memcpy(dst, src, bytes, 0, 0, dst_id, src_id) != 0) {
      fprintf(stderr, "omp_target_memcpy of %zu bytes failed\n", bytes);
      return 1;
    }
    return 0;
  }

  // Set bytes bytes of device memory at dst to the low byte of value with a
  // target loop. Return 0.
  int fill(void* dst, int value, size_t bytes) {
    unsigned char* d = static_cast<unsigned char*>(dst);
    const unsigned char v = static_cast<unsigned char>(value);
    const int64_t n = static_cast<int64_t>(bytes);
    #pragma omp target teams distribute parallel for is_device_ptr(d) \
        device(device_id)
    for (int64_t i = 0; i < n; ++i) d[i] = v;
    return 0;
  }
};

// Drop values's buffer, not just its size, by swapping with an empty vector.
template <typename T>
void release_vector(std::vector<T>& values) {
  std::vector<T> empty;
  values.swap(empty);
}

enum Kind { Invalid = 0, Oob, Unmapped, Overflow, Resident, Hit, Miss };

struct ColSpec {
  Kind kind;
  int32_t token;
  int32_t block;
  int32_t g;
  int32_t resident_row;
};

// Compare n elements with ==. On the first difference, print name, the index,
// and both values cast to long long, then return false. Otherwise return true.
template <typename T>
static bool equal_n(const T* got, const T* ref, size_t n, const char* name) {
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != ref[i]) {
      fprintf(stderr, "mismatch %s at %zu: device %lld ref %lld\n", name, i,
              static_cast<long long>(got[i]), static_cast<long long>(ref[i]));
      return false;
    }
  }
  return true;
}

// Return true when n bytes match. Otherwise find the first differing byte,
// print name, its index, and both bytes as unsigned values, and return false.
// If memcmp reports a difference the scan does not find, print that and
// return false.
static bool equal_bytes(const char* got, const char* ref, size_t n,
                         const char* name) {
  if (std::memcmp(got, ref, n) == 0) return true;
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != ref[i]) {
      fprintf(stderr, "mismatch %s at byte %zu: device %d ref %d\n", name, i,
              static_cast<unsigned char>(got[i]),
              static_cast<unsigned char>(ref[i]));
      return false;
    }
  }
  fprintf(stderr, "mismatch %s: buffers differ\n", name);
  return false;
}

// Fill bytes bytes with tag, then mark both ends of every row's value span
// and scale span so a short or mis-addressed copy is visible. Requires
// value_bytes >= 1 and row_bytes > value_bytes; both ends of each span are
// written. The scale length is row_bytes - value_bytes.
static void paint_split(char* buf, size_t bytes, int64_t rows, int32_t block_size,
                        int64_t block_stride, int64_t row_bytes,
                        int64_t value_bytes, unsigned char tag) {
  std::memset(buf, tag, bytes);
  const int64_t scale_bytes = row_bytes - value_bytes;
  for (int64_t r = 0; r < rows; ++r) {
    char* page = buf + (r / block_size) * block_stride;
    const int64_t off = r % block_size;
    char* value = page + off * value_bytes;
    char* scale = page + block_size * value_bytes + off * scale_bytes;
    value[0] = static_cast<char>(tag ^ static_cast<unsigned char>(r & 255));
    value[value_bytes - 1] = static_cast<char>((r * 3) & 255);
    scale[0] = static_cast<char>(tag ^ static_cast<unsigned char>((r * 5) & 255));
    scale[scale_bytes - 1] = static_cast<char>((r * 7) & 255);
  }
}

// Fill kinds, whose size is top_k. full makes every column Resident.
// Otherwise every column starts as Miss and a prefix is overwritten:
// top_k >= 16 writes one Invalid, Oob, Unmapped, and Overflow, then
// max(1, top_k/8) Resident and max(1, top_k/4) Hit;
// top_k >= 4 writes one Invalid, Resident, and Hit;
// top_k == 3 writes one Resident and one Hit;
// top_k == 2 writes one Resident;
// top_k == 1 stays all Miss.
// A prefix that would pass top_k stops at the end.
static void fill_kinds(std::vector<Kind>& kinds, bool full) {
  const int top_k = static_cast<int>(kinds.size());
  std::fill(kinds.begin(), kinds.end(), Miss);
  if (full) {
    std::fill(kinds.begin(), kinds.end(), Resident);
    return;
  }
  int cursor = 0;
  auto put = [&](Kind kind, int n) {
    for (int c = 0; c < n && cursor < top_k; ++c) kinds[cursor++] = kind;
  };
  if (top_k >= 16) {
    put(Invalid, 1);
    put(Oob, 1);
    put(Unmapped, 1);
    put(Overflow, 1);
    put(Resident, std::max(1, top_k / 8));
    put(Hit, std::max(1, top_k / 4));
  } else if (top_k >= 4) {
    put(Invalid, 1);
    put(Resident, 1);
    put(Hit, 1);
  } else if (top_k == 3) {
    put(Resident, 1);
    put(Hit, 1);
  } else if (top_k == 2) {
    put(Resident, 1);
  }
}

// One column spec per top-k entry. Kinds come from fill_kinds. Invalid and
// Oob stay at token, block, and g of -1. Resident, Hit, Miss, Unmapped, and
// Overflow that are present are packed, in that order, into contiguous token
// ranges aligned up to block_size. g and resident_row stay -1; run() fills
// them from the block tables.
static std::vector<ColSpec> build_columns(int top_k, bool full, int block_size) {
  std::vector<Kind> kinds(static_cast<size_t>(top_k));
  fill_kinds(kinds, full);
  std::vector<ColSpec> cols(static_cast<size_t>(top_k));
  for (int i = 0; i < top_k; ++i) {
    cols[static_cast<size_t>(i)].kind = kinds[static_cast<size_t>(i)];
    cols[static_cast<size_t>(i)].token = -1;
    cols[static_cast<size_t>(i)].block = -1;
    cols[static_cast<size_t>(i)].g = -1;
    cols[static_cast<size_t>(i)].resident_row = -1;
  }
  int32_t next_token = 0;
  auto pack = [&](Kind kind) {
    bool any = false;
    for (int i = 0; i < top_k; ++i) {
      if (kinds[static_cast<size_t>(i)] == kind) any = true;
    }
    if (!any) return;
    const int rem = next_token % block_size;
    if (rem) next_token += block_size - rem;
    for (int i = 0; i < top_k; ++i) {
      if (kinds[static_cast<size_t>(i)] != kind) continue;
      cols[static_cast<size_t>(i)].token = next_token;
      cols[static_cast<size_t>(i)].block = next_token / block_size;
      next_token++;
    }
  };
  pack(Resident);
  pack(Hit);
  pack(Miss);
  pack(Unmapped);
  pack(Overflow);
  return cols;
}

// Check the gather zero-fill branch, which resolve never feeds because it
// turns a host index past the pool into -1. ref_hot and ref_hot_b must
// already match d_hot and d_hot_b. Both references run with one row and
// top_k 1, host index host_rows, and hot row 0, and they modify those
// vectors in place. The device launches keep the real grid: every compact
// count is 0 except row 0, whose one entry is the out-of-range index, and
// every plan miss flag is 0 except index 0. The whole of each hot cache is
// compared, plus plan attention index 0. Returns 0 on agreement and 1 on a
// failed copy, rejected launch, or mismatch. Does not restore the device
// buffers.
static int check_gather_zero_fill(
    OmpArena& arena, char* pinned_host, char* pinned_host_b, char* d_hot, char* d_hot_b,
    std::vector<char>& ref_hot, std::vector<char>& ref_hot_b,
    std::vector<char>& got_hot, int32_t* d_swap_host, int32_t* d_swap_device,
    int32_t* d_swap_counts, int32_t* d_resolved, int32_t* d_hot_idx,
    int32_t* d_miss, int32_t* d_attn, int32_t* d_batch_state, int num_rows,
    int top_k, int64_t host_rows, int64_t hot_rows, int32_t block_size,
    int64_t block_stride, int64_t row_bytes, int64_t value_bytes,
    int64_t attention_block_stride, size_t index_n) {
  const int32_t g = static_cast<int32_t>(host_rows);
  const int32_t dst = 0;

  int32_t swap_host = g;
  int32_t swap_device = dst;
  int32_t swap_count = 1;
  hisparse_gather_compact_reference(
      pinned_host, ref_hot.data(), &swap_host, &swap_device, &swap_count, 1, 1,
      host_rows, block_size, block_stride, hot_rows, row_bytes, value_bytes,
      block_stride, block_size);
  CHECK(arena.fill(d_swap_host, 0xFF, index_n * sizeof(int32_t)));
  CHECK(arena.fill(d_swap_device, 0xFF, index_n * sizeof(int32_t)));
  CHECK(arena.fill(d_swap_counts, 0,
                        static_cast<size_t>(num_rows) * sizeof(int32_t)));
  CHECK(arena.copy(d_swap_host, &swap_host, sizeof(swap_host)));
  CHECK(arena.copy(d_swap_device, &swap_device, sizeof(swap_device)));
  CHECK(arena.copy(d_swap_counts, &swap_count, sizeof(swap_count)));
  CHECK(hisparse_gather_compact_launch(
      pinned_host, d_hot, d_swap_host, d_swap_device, d_swap_counts, host_rows,
      block_size, block_stride, hot_rows, row_bytes, value_bytes, block_stride,
      block_size, num_rows, top_k));
  CHECK(arena.copy(got_hot.data(), d_hot, got_hot.size()));
  if (!equal_bytes(got_hot.data(), ref_hot.data(), ref_hot.size(),
                   "gather_compact zero-fill")) {
    return 1;
  }

  int32_t global = g;
  int32_t hot = dst;
  int32_t miss = 1;
  int32_t attn = -2;
  int32_t batch = 0;
  hisparse_gather_plan_reference(
      pinned_host_b, ref_hot_b.data(), &global, &hot, &miss, &attn, &batch, 1, 1,
      host_rows, block_size, block_stride, hot_rows, row_bytes, value_bytes,
      block_stride, block_size, attention_block_stride);
  CHECK(arena.fill(d_miss, 0, index_n * sizeof(int32_t)));
  CHECK(arena.copy(d_resolved, &global, sizeof(global)));
  CHECK(arena.copy(d_hot_idx, &hot, sizeof(hot)));
  CHECK(arena.copy(d_miss, &miss, sizeof(miss)));
  CHECK(hisparse_gather_plan_launch(
      pinned_host_b, d_hot_b, d_resolved, d_hot_idx, d_miss, d_attn,
      d_batch_state, host_rows, block_size, block_stride, hot_rows, row_bytes,
      value_bytes, block_stride, block_size, num_rows, top_k,
      attention_block_stride));
  CHECK(arena.copy(got_hot.data(), d_hot_b, got_hot.size()));
  int32_t got_attn = 0;
  CHECK(arena.copy(&got_attn, d_attn, sizeof(got_attn)));
  if (got_attn != attn) {
    fprintf(stderr, "mismatch gather_plan zero-fill attention: device %d ref %d\n",
            got_attn, attn);
    return 1;
  }
  if (!equal_bytes(got_hot.data(), ref_hot_b.data(), ref_hot_b.size(),
                   "gather_plan zero-fill")) {
    return 1;
  }
  return 0;
}

// Build one decode step, check it against the CPU reference, then time
// `repeat` device passes with std::chrono. num_rows, top_k, and repeat must
// be positive and top_k must be in [1, 32767]. The last batch row is padding
// when num_rows >= 2, and the first active request is fully resident when at
// least two requests are active. Returns 1 on a bad argument, a rejected
// launch, a failed allocation or copy, a broken setup invariant, or a
// mismatch. On success prints the exercised paths and the
// average milliseconds of the five launches, then returns 0. The caller
// prints PASS or FAIL.
int run(int num_rows, int top_k, int repeat) {
  if (num_rows <= 0 || top_k <= 0 || top_k > 32767 || repeat <= 0) {
    fprintf(stderr,
            "num_rows and repeat must be positive and top_k in [1, 32767]\n");
    return 1;
  }

  const int32_t block_size = 16;
  const int64_t row_bytes = 512;
  const int64_t value_bytes = 384;
  const int32_t hot_size = top_k;
  const int64_t region_stride = hot_size;
  const int64_t attention_block_stride = 64;
  const int64_t input_row_stride = top_k;
  const int64_t attention_row_stride = top_k;
  const int64_t valid_count_stride = 1;
  const int32_t resident_null_block = 0;
  const int64_t block_stride =
      static_cast<int64_t>(block_size) * row_bytes;

  const bool any_pad = num_rows >= 2;
  const int n_active = any_pad ? num_rows - 1 : num_rows;
  const bool any_full = n_active >= 2;
  {
    const int64_t rows_per_request =
        (static_cast<int64_t>(hot_size) + block_size - 1) / block_size *
        block_size;
    const int64_t max_attention =
        (static_cast<int64_t>(n_active) * rows_per_request - 1) / block_size *
            attention_block_stride +
        block_size - 1;
    if (max_attention > std::numeric_limits<int32_t>::max()) {
      fprintf(stderr, "hot cache is too large for the int32 attention ids\n");
      return 1;
    }
  }
  std::vector<int32_t> request_of_batch(static_cast<size_t>(num_rows), -1);
  std::vector<char> batch_is_full(static_cast<size_t>(num_rows), 0);
  std::vector<char> req_is_full(static_cast<size_t>(n_active), 0);
  int active_i = 0;
  for (int b = 0; b < num_rows; ++b) {
    if (any_pad && b == num_rows - 1) continue;
    const int32_t req = n_active - 1 - active_i;
    request_of_batch[static_cast<size_t>(b)] = req;
    if (any_full && active_i == 0) {
      batch_is_full[static_cast<size_t>(b)] = 1;
      req_is_full[static_cast<size_t>(req)] = 1;
    }
    active_i++;
  }

  std::vector<std::vector<ColSpec>> cols(static_cast<size_t>(n_active));
  int32_t source_num_blocks = 1;
  for (int req = 0; req < n_active; ++req) {
    cols[static_cast<size_t>(req)] = build_columns(
        top_k, req_is_full[static_cast<size_t>(req)] != 0, block_size);
    for (const ColSpec& col : cols[static_cast<size_t>(req)]) {
      if (col.block + 1 > source_num_blocks) source_num_blocks = col.block + 1;
    }
  }
  const int64_t oob_token =
      static_cast<int64_t>(source_num_blocks) * block_size;
  if (oob_token > std::numeric_limits<int32_t>::max()) {
    fprintf(stderr, "source token id does not fit in int32\n");
    return 1;
  }
  for (int req = 0; req < n_active; ++req) {
    for (ColSpec& col : cols[static_cast<size_t>(req)]) {
      if (col.kind == Oob) col.token = static_cast<int32_t>(oob_token);
    }
  }

  const int64_t source_bt_stride = source_num_blocks;
  const int64_t resident_bt_stride = source_num_blocks;
  std::vector<int32_t> h_source(
      static_cast<size_t>(n_active) * static_cast<size_t>(source_num_blocks), 0);
  std::vector<int32_t> h_resident = h_source;
  std::vector<char> source_set(h_source.size(), 0);
  std::vector<char> resident_set(h_source.size(), 0);
  int32_t next_src = 1;
  int32_t next_res = 1;
  for (int req = 0; req < n_active; ++req) {
    for (const ColSpec& col : cols[static_cast<size_t>(req)]) {
      if (col.block < 0) continue;
      const size_t idx = static_cast<size_t>(req) * source_num_blocks +
                         static_cast<size_t>(col.block);
      if ((col.kind == Hit || col.kind == Miss) && !source_set[idx]) {
        h_source[idx] = next_src++;
        source_set[idx] = 1;
      }
      if (col.kind == Resident && !resident_set[idx]) {
        h_resident[idx] = next_res++;
        resident_set[idx] = 1;
      }
    }
  }
  const int64_t host_rows = static_cast<int64_t>(next_src) * block_size;
  if (host_rows > std::numeric_limits<int32_t>::max() ||
      host_rows * block_size > std::numeric_limits<int32_t>::max()) {
    fprintf(stderr, "host pool is too large for the int32 slot ids\n");
    return 1;
  }
  for (int req = 0; req < n_active; ++req) {
    for (ColSpec& col : cols[static_cast<size_t>(req)]) {
      if (col.kind != Overflow || col.block < 0) continue;
      const size_t idx = static_cast<size_t>(req) * source_num_blocks +
                         static_cast<size_t>(col.block);
      h_source[idx] = static_cast<int32_t>(host_rows);
    }
  }
  for (int req = 0; req < n_active; ++req) {
    for (ColSpec& col : cols[static_cast<size_t>(req)]) {
      if (col.block < 0 || col.token < 0) continue;
      const size_t idx = static_cast<size_t>(req) * source_num_blocks +
                         static_cast<size_t>(col.block);
      if (col.kind == Hit || col.kind == Miss) {
        const int32_t phys = h_source[idx];
        col.g = phys * block_size + col.token % block_size;
      } else if (col.kind == Resident) {
        const int32_t phys = h_resident[idx];
        col.resident_row = phys * block_size + col.token % block_size;
        col.g = -1;
      }
    }
  }

  const int blocks_per_row = (hot_size + block_size - 1) / block_size;
  const int64_t hot_table_stride = blocks_per_row;
  const int64_t hot_rows =
      static_cast<int64_t>(n_active) * blocks_per_row * block_size;
  const size_t index_n =
      static_cast<size_t>(num_rows) * static_cast<size_t>(top_k);
  const size_t state_n =
      static_cast<size_t>(n_active) * static_cast<size_t>(hot_size);
  std::vector<int32_t> h_state(static_cast<size_t>(n_active) + 2);
  for (int req = 0; req < n_active; ++req) {
    h_state[static_cast<size_t>(req)] = (req + 1) % n_active;
  }
  h_state[static_cast<size_t>(n_active)] = -1;
  h_state[static_cast<size_t>(n_active) + 1] = n_active;
  const int32_t request_state_count = n_active;
  const int64_t num_request_ids = static_cast<int64_t>(n_active) + 2;
  const int64_t num_state_rows = n_active;

  std::vector<int32_t> h_block_table(static_cast<size_t>(n_active) *
                                     static_cast<size_t>(blocks_per_row));
  for (int req = 0; req < n_active; ++req) {
    for (int c = 0; c < blocks_per_row; ++c) {
      h_block_table[static_cast<size_t>(req) * blocks_per_row + c] =
          req * blocks_per_row + c;
    }
  }

  std::vector<int32_t> h_global(index_n, -1);
  std::vector<int32_t> h_dgi(state_n, -1);
  std::vector<int16_t> h_lru(state_n, 0);
  std::vector<int32_t> h_batch_state(static_cast<size_t>(num_rows), 0);
  bool any_resident = false;
  bool any_overflow = false;
  bool any_corrupt = false;
  bool any_request_map = false;
  int n_writes = 0;

  for (int b = 0; b < num_rows; ++b) {
    if (request_of_batch[static_cast<size_t>(b)] < 0) {
      h_batch_state[static_cast<size_t>(b)] = -1;
      continue;
    }
    const int32_t req = request_of_batch[static_cast<size_t>(b)];
    if (req != b) any_request_map = true;
    const int32_t state = h_state[static_cast<size_t>(req)];
    const bool full = batch_is_full[static_cast<size_t>(b)] != 0;
    int n_hit = 0;
    int n_miss = 0;
    int n_phase1 = 0;
    for (const ColSpec& col : cols[static_cast<size_t>(req)]) {
      if (col.kind == Hit) n_hit++;
      else if (col.kind == Miss) n_miss++;
      else n_phase1++;
      if (col.kind == Resident) any_resident = true;
      if (col.kind == Overflow) any_overflow = true;
    }
    const bool corrupt = !full && n_phase1 > 0;
    if (corrupt) any_corrupt = true;
    const int real_slots = corrupt ? hot_size - 1 : hot_size;
    if (n_hit + n_miss > real_slots) {
      fprintf(stderr, "not enough hot slots for hits and misses\n");
      return 1;
    }
    // A corrupt row puts -1 at the oldest LRU position, so the first miss
    // is handed the out-of-range slot.
    for (int s = 0; s < hot_size; ++s) {
      const int16_t slot = !corrupt ? static_cast<int16_t>(s)
                           : s == 0 ? static_cast<int16_t>(-1)
                                    : static_cast<int16_t>(s - 1);
      h_lru[static_cast<size_t>(state) * hot_size + s] = slot;
    }
    int hit_i = 0;
    int miss_i = 0;
    for (int i = 0; i < top_k; ++i) {
      const ColSpec& col = cols[static_cast<size_t>(req)][static_cast<size_t>(i)];
      h_global[static_cast<size_t>(b) * top_k + i] = col.token;
      if (col.kind == Hit) {
        h_dgi[static_cast<size_t>(state) * hot_size + n_miss + hit_i] = col.g;
        hit_i++;
      } else if (col.kind == Miss) {
        h_dgi[static_cast<size_t>(state) * hot_size + miss_i] = col.g;
        miss_i++;
        n_writes++;
      }
    }
  }

  std::vector<int64_t> h_src_index(static_cast<size_t>(n_writes) + 3);
  std::vector<int64_t> h_backup_dst(static_cast<size_t>(n_writes) + 3);
  std::vector<int32_t> h_req(static_cast<size_t>(n_writes) + 4);
  std::vector<int64_t> h_written(static_cast<size_t>(n_writes) + 4);
  int write_i = 0;
  for (int b = 0; b < num_rows; ++b) {
    const int32_t req = request_of_batch[static_cast<size_t>(b)];
    if (req < 0) continue;
    for (const ColSpec& col : cols[static_cast<size_t>(req)]) {
      if (col.kind != Miss) continue;
      h_src_index[static_cast<size_t>(write_i)] = write_i;
      h_backup_dst[static_cast<size_t>(write_i)] = col.g;
      h_req[static_cast<size_t>(write_i)] = req;
      h_written[static_cast<size_t>(write_i)] = col.g;
      write_i++;
    }
  }
  const int32_t src_rows = std::max(write_i, 1);
  const int32_t num_backup = write_i + 3;
  const int32_t num_invalidate = write_i + 4;
  h_src_index[static_cast<size_t>(write_i)] = -1;
  h_backup_dst[static_cast<size_t>(write_i)] = 0;
  h_src_index[static_cast<size_t>(write_i) + 1] = 0;
  h_backup_dst[static_cast<size_t>(write_i) + 1] = host_rows;
  h_src_index[static_cast<size_t>(write_i) + 2] = src_rows;
  h_backup_dst[static_cast<size_t>(write_i) + 2] = 0;
  h_req[static_cast<size_t>(write_i)] = -1;
  h_written[static_cast<size_t>(write_i)] = 1;
  h_req[static_cast<size_t>(write_i) + 1] = n_active;
  h_written[static_cast<size_t>(write_i) + 1] = 1;
  h_req[static_cast<size_t>(write_i) + 2] = n_active + 1;
  h_written[static_cast<size_t>(write_i) + 2] = 1;
  h_req[static_cast<size_t>(write_i) + 3] = n_active + 2;
  h_written[static_cast<size_t>(write_i) + 3] = 1;

  const int64_t host_bytes = cache_bytes(host_rows, block_size, block_stride);
  const int64_t hot_bytes = cache_bytes(hot_rows, block_size, block_stride);
  const int64_t src_bytes = cache_bytes(src_rows, block_size, block_stride);
  OmpArena arena;
  char* pinned_host = nullptr;
  char* pinned_host_b = nullptr;
  CHECK(arena.alloc_pinned(pinned_host, static_cast<size_t>(host_bytes)));
  CHECK(arena.alloc_pinned(pinned_host_b, static_cast<size_t>(host_bytes)));
  paint_split(pinned_host, static_cast<size_t>(host_bytes), host_rows,
              block_size, block_stride, row_bytes, value_bytes, 0x11);
  paint_split(pinned_host_b, static_cast<size_t>(host_bytes), host_rows,
              block_size, block_stride, row_bytes, value_bytes, 0x22);
  std::vector<char> h_src(static_cast<size_t>(src_bytes));
  std::vector<char> h_hot(static_cast<size_t>(hot_bytes));
  std::vector<char> h_hot_b(static_cast<size_t>(hot_bytes));
  paint_split(h_src.data(), h_src.size(), src_rows, block_size, block_stride,
              row_bytes, value_bytes, 0x33);
  paint_split(h_hot.data(), h_hot.size(), hot_rows, block_size, block_stride,
              row_bytes, value_bytes, 0xA5);
  paint_split(h_hot_b.data(), h_hot_b.size(), hot_rows, block_size,
              block_stride, row_bytes, value_bytes, 0x3C);

  printf("hisparse rows=%d top_k=%d active=%d writes=%d host_rows=%lld "
         "hot_rows=%lld row_bytes=%lld value_bytes=%lld repeat=%d\n",
         num_rows, top_k, n_active, write_i, static_cast<long long>(host_rows),
         static_cast<long long>(hot_rows), static_cast<long long>(row_bytes),
         static_cast<long long>(value_bytes), repeat);
  fflush(stdout);

  std::vector<int32_t> ref_dgi = h_dgi;
  std::vector<int16_t> ref_lru = h_lru;
  std::vector<char> ref_host(pinned_host, pinned_host + host_bytes);
  hisparse_backup_reference(
      h_src.data(), ref_host.data(), h_src_index.data(), h_backup_dst.data(),
      num_backup, src_rows, host_rows, block_size, block_stride, block_size,
      block_stride, row_bytes, value_bytes);
  if (write_i > 0 &&
      std::memcmp(ref_host.data(), pinned_host, ref_host.size()) == 0) {
    fprintf(stderr, "backup reference did not change the host pool\n");
    return 1;
  }
  hisparse_invalidate_reference(ref_dgi.data(), h_state.data(), h_req.data(),
                                h_written.data(), num_invalidate,
                                num_request_ids, num_state_rows, region_stride);
  const std::vector<int32_t> ref_dgi_invalidated = ref_dgi;

  std::vector<int32_t> ref_hot_idx(index_n, -2);
  std::vector<int32_t> ref_attn(index_n, -2);
  std::vector<int32_t> ref_miss(index_n, 0);
  std::vector<int32_t> ref_resolved(index_n, -2);
  std::vector<int32_t> ref_valid(static_cast<size_t>(num_rows), -1);
  std::vector<int32_t> ref_swap_host(index_n, -2);
  std::vector<int32_t> ref_swap_device(index_n, -2);
  std::vector<int32_t> ref_swap_counts(static_cast<size_t>(num_rows), -1);
  unsigned long long ref_stats[2] = {0, 0};
  if (!hisparse_resolve_reference(
          h_block_table.data(), h_global.data(), request_of_batch.data(),
          h_source.data(), h_resident.data(), ref_hot_idx.data(),
          ref_attn.data(), ref_miss.data(), ref_resolved.data(),
          ref_valid.data(), ref_swap_host.data(), ref_swap_device.data(),
          ref_swap_counts.data(), ref_dgi.data(), ref_lru.data(),
          h_state.data(), request_state_count, ref_stats, num_rows, top_k,
          hot_size, host_rows, hot_table_stride, block_size, region_stride,
          attention_block_stride, source_bt_stride, n_active, source_num_blocks,
          block_size, resident_bt_stride, n_active, source_num_blocks,
          block_size, resident_null_block, input_row_stride,
          attention_row_stride, valid_count_stride)) {
    return 1;
  }
  if (ref_stats[1] == 0) {
    fprintf(stderr, "setup produced no misses\n");
    return 1;
  }
  if (top_k >= 3 && ref_stats[0] == 0) {
    fprintf(stderr, "setup produced no LRU hits\n");
    return 1;
  }
  bool any_tripwire = false;
  for (int b = 0; b < num_rows && !any_tripwire; ++b) {
    for (int m = 0; m < ref_swap_counts[static_cast<size_t>(b)]; ++m) {
      if (ref_swap_device[static_cast<size_t>(b) * top_k + m] < 0) {
        any_tripwire = true;
        break;
      }
    }
  }
  if (any_corrupt && !any_tripwire) {
    fprintf(stderr, "corrupt LRU slot was never handed to a miss\n");
    return 1;
  }
  const std::vector<int32_t> ref_attn_resolve = ref_attn;

  std::vector<char> ref_hot = h_hot;
  hisparse_gather_compact_reference(
      ref_host.data(), ref_hot.data(), ref_swap_host.data(),
      ref_swap_device.data(), ref_swap_counts.data(), num_rows, top_k, host_rows,
      block_size, block_stride, hot_rows, row_bytes, value_bytes, block_stride,
      block_size);
  std::vector<char> ref_hot_b = h_hot_b;
  hisparse_gather_plan_reference(
      pinned_host_b, ref_hot_b.data(), ref_resolved.data(), ref_hot_idx.data(),
      ref_miss.data(), ref_attn.data(), h_batch_state.data(), num_rows, top_k,
      host_rows, block_size, block_stride, hot_rows, row_bytes, value_bytes,
      block_stride, block_size, attention_block_stride);

  int32_t *d_block_table = nullptr, *d_global = nullptr, *d_dgi = nullptr;
  int32_t *d_dgi_init = nullptr, *d_hot_idx = nullptr, *d_attn = nullptr;
  int32_t *d_miss = nullptr, *d_resolved = nullptr, *d_valid = nullptr;
  int32_t *d_swap_host = nullptr, *d_swap_device = nullptr;
  int32_t *d_swap_counts = nullptr, *d_state = nullptr, *d_req = nullptr;
  int32_t *d_request = nullptr, *d_source = nullptr, *d_resident = nullptr;
  int32_t *d_batch_state = nullptr;
  int16_t *d_lru = nullptr, *d_lru_init = nullptr;
  int64_t *d_src_index = nullptr, *d_backup_dst = nullptr, *d_written = nullptr;
  unsigned long long* d_stats = nullptr;
  char *d_src = nullptr, *d_hot = nullptr, *d_hot_b = nullptr;

  CHECK(arena.alloc_device(d_block_table, h_block_table.size()));
  CHECK(arena.alloc_device(d_global, index_n));
  CHECK(arena.alloc_device(d_dgi, state_n));
  CHECK(arena.alloc_device(d_dgi_init, state_n));
  CHECK(arena.alloc_device(d_lru, state_n));
  CHECK(arena.alloc_device(d_lru_init, state_n));
  CHECK(arena.alloc_device(d_hot_idx, index_n));
  CHECK(arena.alloc_device(d_attn, index_n));
  CHECK(arena.alloc_device(d_miss, index_n));
  CHECK(arena.alloc_device(d_resolved, index_n));
  CHECK(arena.alloc_device(d_valid, static_cast<size_t>(num_rows)));
  CHECK(arena.alloc_device(d_swap_host, index_n));
  CHECK(arena.alloc_device(d_swap_device, index_n));
  CHECK(arena.alloc_device(d_swap_counts, static_cast<size_t>(num_rows)));
  CHECK(arena.alloc_device(d_state, h_state.size()));
  CHECK(arena.alloc_device(d_req, h_req.size()));
  CHECK(arena.alloc_device(d_request, request_of_batch.size()));
  CHECK(arena.alloc_device(d_source, h_source.size()));
  CHECK(arena.alloc_device(d_resident, h_resident.size()));
  CHECK(arena.alloc_device(d_batch_state, h_batch_state.size()));
  CHECK(arena.alloc_device(d_src_index, h_src_index.size()));
  CHECK(arena.alloc_device(d_backup_dst, h_backup_dst.size()));
  CHECK(arena.alloc_device(d_written, h_written.size()));
  CHECK(arena.alloc_device(d_stats, 2));
  CHECK(arena.alloc_device(d_src, h_src.size()));
  CHECK(arena.alloc_device(d_hot, h_hot.size()));
  CHECK(arena.alloc_device(d_hot_b, h_hot_b.size()));

  CHECK(arena.copy(d_block_table, h_block_table.data(),
                        h_block_table.size() * sizeof(int32_t)));
  CHECK(arena.copy(d_global, h_global.data(), index_n * sizeof(int32_t)));
  CHECK(arena.copy(d_dgi, h_dgi.data(), state_n * sizeof(int32_t)));
  CHECK(arena.copy(d_dgi_init, h_dgi.data(), state_n * sizeof(int32_t)));
  CHECK(arena.copy(d_lru, h_lru.data(), state_n * sizeof(int16_t)));
  CHECK(arena.copy(d_lru_init, h_lru.data(), state_n * sizeof(int16_t)));
  CHECK(arena.copy(d_state, h_state.data(), h_state.size() * sizeof(int32_t)));
  CHECK(arena.copy(d_req, h_req.data(), h_req.size() * sizeof(int32_t)));
  CHECK(arena.copy(d_request, request_of_batch.data(),
                        request_of_batch.size() * sizeof(int32_t)));
  CHECK(arena.copy(d_source, h_source.data(),
                        h_source.size() * sizeof(int32_t)));
  CHECK(arena.copy(d_resident, h_resident.data(),
                        h_resident.size() * sizeof(int32_t)));
  CHECK(arena.copy(d_batch_state, h_batch_state.data(),
                        h_batch_state.size() * sizeof(int32_t)));
  CHECK(arena.copy(d_src_index, h_src_index.data(),
                        h_src_index.size() * sizeof(int64_t)));
  CHECK(arena.copy(d_backup_dst, h_backup_dst.data(),
                        h_backup_dst.size() * sizeof(int64_t)));
  CHECK(arena.copy(d_written, h_written.data(),
                        h_written.size() * sizeof(int64_t)));
  CHECK(arena.copy(d_src, h_src.data(), h_src.size()));
  CHECK(arena.copy(d_hot, h_hot.data(), h_hot.size()));
  CHECK(arena.copy(d_hot_b, h_hot_b.data(), h_hot_b.size()));
  std::vector<int32_t> h_out_init(index_n, -2);
  CHECK(arena.copy(d_swap_host, h_out_init.data(), index_n * sizeof(int32_t)));
  CHECK(arena.copy(d_swap_device, h_out_init.data(),
                        index_n * sizeof(int32_t)));
  CHECK(arena.copy(d_hot_idx, h_out_init.data(), index_n * sizeof(int32_t)));
  CHECK(arena.copy(d_attn, h_out_init.data(), index_n * sizeof(int32_t)));
  CHECK(arena.copy(d_resolved, h_out_init.data(), index_n * sizeof(int32_t)));
  CHECK(arena.fill(d_miss, 0, index_n * sizeof(int32_t)));
  CHECK(arena.fill(d_stats, 0, 2 * sizeof(unsigned long long)));

  HisparseResolveArgs args{};
  args.hot_block_table = d_block_table;
  args.global_indices = d_global;
  args.request_ids = d_request;
  args.source_block_table = d_source;
  args.resident_block_table = d_resident;
  args.resolved_global_indices = d_resolved;
  args.valid_counts = d_valid;
  args.swap_host_physical_rows = d_swap_host;
  args.swap_device_physical_rows = d_swap_device;
  args.swap_counts = d_swap_counts;
  args.hot_indices = d_hot_idx;
  args.attention_indices = d_attn;
  args.miss_mask = d_miss;
  args.device_global_indices = d_dgi;
  args.lru_slots = d_lru;
  args.stats = d_stats;
  args.request_state_indices = d_state;
  args.request_state_count = request_state_count;
  args.host_rows = host_rows;
  args.hot_table_stride = hot_table_stride;
  args.hot_block_size = block_size;
  args.top_k = top_k;
  args.hot_size = hot_size;
  args.region_stride = region_stride;
  args.attention_block_stride = attention_block_stride;
  args.source_bt_stride = source_bt_stride;
  args.source_num_reqs = n_active;
  args.source_num_blocks = source_num_blocks;
  args.source_block_size = block_size;
  args.resident_bt_stride = resident_bt_stride;
  args.resident_num_reqs = n_active;
  args.resident_num_blocks = source_num_blocks;
  args.resident_block_size = block_size;
  args.resident_null_block = resident_null_block;
  args.input_row_stride = input_row_stride;
  args.attention_row_stride = attention_row_stride;
  args.valid_count_stride = valid_count_stride;
  args.num_rows = num_rows;
  const HisparseResolveScratchBytes scratch =
      hisparse_resolve_scratch_bytes(num_rows, top_k, hot_size);
  CHECK(arena.alloc_device(args.scratch_topk, scratch.topk / sizeof(int32_t)));
  CHECK(arena.alloc_device(args.scratch_hash_keys,
                           scratch.hash_keys / sizeof(int32_t)));
  CHECK(arena.alloc_device(args.scratch_hash_vals,
                           scratch.hash_vals / sizeof(int16_t)));
  CHECK(arena.alloc_device(args.scratch_lru_out,
                           scratch.lru_out / sizeof(int16_t)));
  CHECK(arena.alloc_device(args.scratch_found,
                           scratch.found / sizeof(int16_t)));
  CHECK(arena.alloc_device(args.scratch_team, scratch.team / sizeof(int32_t)));

  CHECK(hisparse_backup_launch(
      d_src, d_src_index, pinned_host, d_backup_dst, row_bytes, value_bytes,
      block_stride, block_size, block_stride, block_size, num_backup, src_rows,
      host_rows));
  if (!equal_bytes(pinned_host, ref_host.data(), ref_host.size(),
                   "backup host cache")) {
    return 1;
  }

  CHECK(hisparse_invalidate_written_slots_launch(
      d_dgi, d_state, d_req, d_written, num_invalidate, num_request_ids,
      num_state_rows, region_stride));
  std::vector<int32_t> got_dgi(state_n);
  CHECK(arena.copy(got_dgi.data(), d_dgi, state_n * sizeof(int32_t)));
  if (!equal_n(got_dgi.data(), ref_dgi_invalidated.data(), state_n,
               "invalidate device_global_indices")) {
    return 1;
  }

  CHECK(hisparse_resolve_residency_launch(args));
  std::vector<int32_t> got_hot_idx(index_n), got_attn(index_n), got_miss(index_n);
  std::vector<int32_t> got_resolved(index_n),
      got_valid(static_cast<size_t>(num_rows));
  std::vector<int32_t> got_swap_host(index_n), got_swap_device(index_n);
  std::vector<int32_t> got_swap_counts(static_cast<size_t>(num_rows));
  std::vector<int16_t> got_lru(state_n);
  unsigned long long got_stats[2] = {0, 0};
  CHECK(arena.copy(got_hot_idx.data(), d_hot_idx, index_n * sizeof(int32_t)));
  CHECK(arena.copy(got_attn.data(), d_attn, index_n * sizeof(int32_t)));
  CHECK(arena.copy(got_miss.data(), d_miss, index_n * sizeof(int32_t)));
  CHECK(arena.copy(got_resolved.data(), d_resolved, index_n * sizeof(int32_t)));
  CHECK(arena.copy(got_valid.data(), d_valid,
                        static_cast<size_t>(num_rows) * sizeof(int32_t)));
  CHECK(arena.copy(got_swap_host.data(), d_swap_host, index_n * sizeof(int32_t)));
  CHECK(arena.copy(got_swap_device.data(), d_swap_device,
                        index_n * sizeof(int32_t)));
  CHECK(arena.copy(got_swap_counts.data(), d_swap_counts,
                        static_cast<size_t>(num_rows) * sizeof(int32_t)));
  CHECK(arena.copy(got_dgi.data(), d_dgi, state_n * sizeof(int32_t)));
  CHECK(arena.copy(got_lru.data(), d_lru, state_n * sizeof(int16_t)));
  CHECK(arena.copy(got_stats, d_stats, sizeof(got_stats)));
  if (!equal_n(got_hot_idx.data(), ref_hot_idx.data(), index_n, "hot_indices") ||
      !equal_n(got_attn.data(), ref_attn_resolve.data(), index_n,
               "attention_indices") ||
      !equal_n(got_miss.data(), ref_miss.data(), index_n, "miss_mask") ||
      !equal_n(got_resolved.data(), ref_resolved.data(), index_n, "resolved") ||
      !equal_n(got_valid.data(), ref_valid.data(), num_rows, "valid_counts") ||
      !equal_n(got_swap_host.data(), ref_swap_host.data(), index_n, "swap_host") ||
      !equal_n(got_swap_device.data(), ref_swap_device.data(), index_n,
               "swap_device") ||
      !equal_n(got_swap_counts.data(), ref_swap_counts.data(), num_rows,
               "swap_counts") ||
      !equal_n(got_dgi.data(), ref_dgi.data(), state_n, "device_global_indices") ||
      !equal_n(got_lru.data(), ref_lru.data(), state_n, "lru_slots") ||
      !equal_n(got_stats, ref_stats, 2, "stats")) {
    return 1;
  }

  CHECK(hisparse_gather_compact_launch(
      pinned_host, d_hot, d_swap_host, d_swap_device, d_swap_counts, host_rows,
      block_size, block_stride, hot_rows, row_bytes, value_bytes, block_stride,
      block_size, num_rows, top_k));
  CHECK(arena.copy(h_hot.data(), d_hot, h_hot.size()));
  if (!equal_bytes(h_hot.data(), ref_hot.data(), h_hot.size(),
                   "gather_compact hot cache")) {
    return 1;
  }

  CHECK(hisparse_gather_plan_launch(
      pinned_host_b, d_hot_b, d_resolved, d_hot_idx, d_miss, d_attn,
      d_batch_state, host_rows, block_size, block_stride, hot_rows, row_bytes,
      value_bytes, block_stride, block_size, num_rows, top_k,
      attention_block_stride));
  CHECK(arena.copy(h_hot_b.data(), d_hot_b, h_hot_b.size()));
  CHECK(arena.copy(got_attn.data(), d_attn, index_n * sizeof(int32_t)));
  if (!equal_bytes(h_hot_b.data(), ref_hot_b.data(), h_hot_b.size(),
                   "gather_plan hot cache") ||
      !equal_n(got_attn.data(), ref_attn.data(), index_n,
               "gather_plan attention_indices")) {
    return 1;
  }
  if (check_gather_zero_fill(
          arena, pinned_host, pinned_host_b, d_hot, d_hot_b, ref_hot, ref_hot_b, h_hot,
          d_swap_host, d_swap_device, d_swap_counts, d_resolved, d_hot_idx,
          d_miss, d_attn, d_batch_state, num_rows, top_k, host_rows, hot_rows,
          block_size, block_stride, row_bytes, value_bytes,
          attention_block_stride, index_n) != 0) {
    return 1;
  }
  release_vector(ref_host);
  release_vector(ref_hot);
  release_vector(ref_hot_b);
  release_vector(h_hot);
  release_vector(h_hot_b);
  release_vector(h_src);

  double total_ms = 0.0;
  for (int k = 0; k < repeat; ++k) {
    CHECK(arena.copy(d_dgi, d_dgi_init, state_n * sizeof(int32_t)));
    CHECK(arena.copy(d_lru, d_lru_init, state_n * sizeof(int16_t)));
    CHECK(arena.fill(d_stats, 0, sizeof(got_stats)));
    const auto start = std::chrono::steady_clock::now();
    CHECK(hisparse_backup_launch(
        d_src, d_src_index, pinned_host, d_backup_dst, row_bytes, value_bytes,
        block_stride, block_size, block_stride, block_size, num_backup, src_rows,
        host_rows));
    CHECK(hisparse_invalidate_written_slots_launch(
        d_dgi, d_state, d_req, d_written, num_invalidate, num_request_ids,
        num_state_rows, region_stride));
    CHECK(hisparse_resolve_residency_launch(args));
    CHECK(hisparse_gather_compact_launch(
        pinned_host, d_hot, d_swap_host, d_swap_device, d_swap_counts, host_rows,
        block_size, block_stride, hot_rows, row_bytes, value_bytes, block_stride,
        block_size, num_rows, top_k));
    CHECK(hisparse_gather_plan_launch(
        pinned_host_b, d_hot_b, d_resolved, d_hot_idx, d_miss, d_attn,
        d_batch_state, host_rows, block_size, block_stride, hot_rows, row_bytes,
        value_bytes, block_stride, block_size, num_rows, top_k,
        attention_block_stride));
    const auto end = std::chrono::steady_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(end - start).count();
  }

  CHECK(arena.copy(got_dgi.data(), d_dgi, state_n * sizeof(int32_t)));
  CHECK(arena.copy(got_lru.data(), d_lru, state_n * sizeof(int16_t)));
  CHECK(arena.copy(got_stats, d_stats, sizeof(got_stats)));
  if (!equal_n(got_dgi.data(), ref_dgi.data(), state_n,
               "timed device_global_indices") ||
      !equal_n(got_lru.data(), ref_lru.data(), state_n, "timed lru_slots") ||
      !equal_n(got_stats, ref_stats, 2, "timed stats")) {
    return 1;
  }

  printf("checked backup, invalidate, resolve, gather_compact, gather_plan\n");
  printf("paths:");
  printf(" request-ids");
  if (n_active >= 2) printf(" state-map");
  printf(" source-blocks");
  if (any_request_map) printf(" request-map");
  if (any_resident) printf(" resident-rows");
  if (any_pad) printf(" padding");
  if (any_full) printf(" fully-resident");
  if (any_overflow) printf(" host-overflow");
  printf(" zero-fill");
  if (any_corrupt) printf(" corrupt-lru");
  printf(" split-kv\n");
  printf("Average execution time of kernels: %f (ms)\n", total_ms / repeat);
  return 0;
}

// Parse text as base-10 into *out. The whole string must be consumed, errno
// must stay clear, and the value must fit in int. Leading whitespace is
// accepted because strtol skips it. Returns false otherwise.
static bool parse_int(const char* text, int* out) {
  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

// Require exactly three integer arguments: num_rows, top_k, repeat. A wrong
// count or a bad integer prints the usage line and returns 1, with no
// PASS/FAIL. Otherwise print the OpenMP device count, run the benchmark on
// the default device, print PASS or FAIL, and return run's status.
int main(int argc, char** argv) {
  int num_rows = 0, top_k = 0, repeat = 0;
  if (argc != 4 || !parse_int(argv[1], &num_rows) ||
      !parse_int(argv[2], &top_k) || !parse_int(argv[3], &repeat)) {
    printf("Usage: %s <num_rows> <top_k> <repeat>\n", argv[0]);
    return 1;
  }
  printf("OpenMP devices: %d, default device: %d\n", omp_get_num_devices(),
         omp_get_default_device());
  const int status = run(num_rows, top_k, repeat);
  printf("%s\n", status == 0 ? "PASS" : "FAIL");
  return status;
}
