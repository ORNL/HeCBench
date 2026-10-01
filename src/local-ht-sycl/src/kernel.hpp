#pragma once
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <sycl/sycl.hpp>

#define EMPTY 0xFFFFFFFF

#ifdef DEBUG_GPU
#define DEBUG_PRINT_GPU 1
#endif

#ifdef DEBUG_CPU
#define DEBUG_PRINT_CPU 1
#endif

#define LASSM_MIN_QUAL 10
#define LASSM_MIN_HI_QUAL 20
#define LASSM_MIN_VIABLE_DEPTH 0.2
#define LASSM_MIN_EXPECTED_DEPTH 0.3
#define LASSM_RATING_THRES 0
#define LASSM_MIN_KMER_LEN 21
#define LASSM_SHIFT_SIZE 8
#define LASSM_MAX_KMER_LEN 121
#define FULL_MASK 0xffffffff

template<typename T>
inline T atomicAdd(T *addr, T val) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::generic_space> atm(*addr);
  return atm.fetch_add(val);
}

// Claim a hash slot by atomically installing a pointer into an empty (null) slot.
// Returns the previous pointer value (null if the claim succeeded).
inline uintptr_t atomicCAS(char **addr, uintptr_t expected, uintptr_t desired) {
  auto *word = reinterpret_cast<uintptr_t *>(addr);
  sycl::atomic_ref<uintptr_t, sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::generic_space> atm(*word);
  atm.compare_exchange_strong(expected, desired);
  return expected;
}

struct cstr_type{
  char* start_ptr;
  int length;
  cstr_type(){}
  cstr_type(char* ptr, int len){ start_ptr = ptr; length = len; }
  bool operator==(const cstr_type& in2) const{
    bool str_eq = true;
    if(length != EMPTY && in2.length != EMPTY)
      for(int i = 0; i < in2.length; i++){
        if(start_ptr[i] != in2.start_ptr[i]){ str_eq = false; break; }
      }
    return (str_eq && (length == in2.length));
  }
};

inline void print_mer(cstr_type& mer){
  for(int i = 0; i < mer.length; i++) printf("%c", mer.start_ptr[i]);
  printf("\n");
}

inline void cstr_copy(cstr_type& str1, cstr_type& str2){
  for(int i = 0; i < str2.length; i++) str1.start_ptr[i] = str2.start_ptr[i];
  str1.length = str2.length;
}

struct ExtCounts {
  uint32_t count_A;
  uint32_t count_C;
  uint32_t count_G;
  uint32_t count_T;
  void print(){ printf("count_A:%d, count_C:%d, count_G:%d, count_T:%d\n", count_A, count_C, count_G, count_T); }
  void inc(char ext, int count) {
    switch (ext) {
      case 'A': atomicAdd(&count_A, (uint32_t)count); break;
      case 'C': atomicAdd(&count_C, (uint32_t)count); break;
      case 'G': atomicAdd(&count_G, (uint32_t)count); break;
      case 'T': atomicAdd(&count_T, (uint32_t)count); break;
    }
  }
};

struct MerBase {
  char base;
  uint32_t nvotes_hi_q, nvotes, rating;
  void print(){ printf("base:%c, nvotes_hiq_q:%d, nvotes:%d, rating:%d\n", base, nvotes_hi_q, nvotes, rating); }
  uint16_t get_base_rating(int depth) {
    double min_viable = sycl::fmax(LASSM_MIN_VIABLE_DEPTH * depth, 2.0);
    double min_expected_depth = sycl::fmax(LASSM_MIN_EXPECTED_DEPTH * depth, 2.0);
    if (nvotes == 0) return 0;
    if (nvotes == 1) return 1;
    if (nvotes < min_viable) return 2;
    if (min_expected_depth > nvotes && nvotes >= min_viable && nvotes_hi_q < min_viable) return 3;
    if (min_expected_depth > nvotes && nvotes >= min_viable && nvotes_hi_q >= min_viable) return 4;
    if (nvotes >= min_expected_depth && nvotes_hi_q < min_viable) return 5;
    if (nvotes >= min_expected_depth && min_viable < nvotes_hi_q && nvotes_hi_q < min_expected_depth) return 6;
    return 7;
  }
};

struct MerFreqs {
  ExtCounts hi_q_exts, low_q_exts;
  char ext;
  int count;
  bool comp_merbase(MerBase& elem1, MerBase& elem2){
    if(elem1.rating != elem2.rating) return elem1.rating > elem2.rating;
    if (elem1.nvotes_hi_q != elem2.nvotes_hi_q) return elem1.nvotes_hi_q > elem2.nvotes_hi_q;
    if (elem1.nvotes != elem2.nvotes) return elem1.nvotes > elem2.nvotes;
    return true;
  }
  void sort_merbase(MerBase (&merbases)[4]){
    for(int i = 0; i < 4; i++) for(int j = 0; j < 4; j++) if(comp_merbase(merbases[i], merbases[j])){
      MerBase temp = merbases[i]; merbases[i] = merbases[j]; merbases[j] = temp;
    }
  }
  void set_ext(int seq_depth) {
    MerBase mer_bases[4] = {{'A', hi_q_exts.count_A, low_q_exts.count_A, 0},
                            {'C', hi_q_exts.count_C, low_q_exts.count_C, 0},
                            {'G', hi_q_exts.count_G, low_q_exts.count_G, 0},
                            {'T', hi_q_exts.count_T, low_q_exts.count_T, 0}};
    for (int i = 0; i < 4; i++) mer_bases[i].rating = mer_bases[i].get_base_rating(seq_depth);
    sort_merbase(mer_bases);
    int top_rating = mer_bases[0].rating;
    int runner_up_rating = mer_bases[1].rating;
    // Preserve CUDA's non-fatal assertion behavior without device printf.
    int top_rated_base = mer_bases[0].base;
    ext = 'X'; count = 0;
    if (top_rating > LASSM_RATING_THRES) {
      if (top_rating <= 3) { if (runner_up_rating == 0) ext = top_rated_base; }
      else if (top_rating < 6) { if (runner_up_rating < 3) ext = top_rated_base; }
      else if (top_rating == 6) { if (runner_up_rating < 4) ext = top_rated_base; }
      else {
        if (runner_up_rating < 7) ext = top_rated_base;
        else {
          if (mer_bases[2].rating == 7 || mer_bases[0].nvotes == mer_bases[1].nvotes) ext = 'F';
          else if (mer_bases[0].nvotes > mer_bases[1].nvotes) ext = mer_bases[0].base;
          else if (mer_bases[1].nvotes > mer_bases[0].nvotes) ext = mer_bases[1].base;
        }
      }
    }
    for (int i = 0; i < 4; i++) if (mer_bases[i].base == ext) { count = mer_bases[i].nvotes; break; }
  }
};

struct loc_ht{ cstr_type key; MerFreqs val; loc_ht(){} loc_ht(cstr_type in_key, MerFreqs in_val){ key = in_key; val = in_val; } };
struct loc_ht_bool{ cstr_type key; bool val; loc_ht_bool(){} loc_ht_bool(cstr_type in_key, bool in_val){ key = in_key; val = in_val; } };

inline int bcast_warp(sycl::sub_group sg, int arg) {
  int value = arg;
  return sycl::select_from_group(sg, value, 0);
}

inline unsigned hash_func(cstr_type key, uint32_t max_size){
  unsigned hash, i;
  for(hash = i = 0; i < key.length; ++i) { hash += key.start_ptr[i]; hash += (hash << 10); hash ^= (hash >> 6); }
  hash += (hash << 3); hash ^= (hash >> 11); hash += (hash << 15);
  return hash % max_size;
}
#define MIX(h,k,m) { k *= m; k ^= k >> r; k *= m; h *= m; h ^= k; }

inline uint32_t MurmurHashAligned2 (cstr_type key_in, uint32_t max_size) {
  int len = key_in.length;
  char* key = key_in.start_ptr;
  const uint32_t m = 0x5bd1e995;
  const int r = 24;
  uint32_t seed = 0x3FB0BB5F;
  const unsigned char * data = (const unsigned char *)key;
  uint32_t h = seed ^ len;
  int align = (uint64_t)data & 3;
  if(align && (len >= 4)) {
    uint32_t t = 0, d = 0;
    switch(align) { case 1: t |= data[2] << 16; [[fallthrough]]; case 2: t |= data[1] << 8; [[fallthrough]]; case 3: t |= data[0]; }
    t <<= (8 * align); data += 4-align; len -= 4-align;
    int sl = 8 * (4-align); int sr = 8 * align;
    while(len >= 4) { d = *(uint32_t *)data; t = (t >> sr) | (d << sl); uint32_t k = t; MIX(h,k,m); t = d; data += 4; len -= 4; }
    d = 0;
    if(len >= align) {
      switch(align) { case 3: d |= data[2] << 16; [[fallthrough]]; case 2: d |= data[1] << 8; [[fallthrough]]; case 1: d |= data[0]; }
      uint32_t k = (t >> sr) | (d << sl); MIX(h,k,m); data += align; len -= align;
      switch(len) { case 3: h ^= data[2] << 16; [[fallthrough]]; case 2: h ^= data[1] << 8; [[fallthrough]]; case 1: h ^= data[0]; h *= m; }
    } else {
      switch(len) { case 3: d |= data[2] << 16; [[fallthrough]]; case 2: d |= data[1] << 8; [[fallthrough]]; case 1: d |= data[0]; [[fallthrough]]; case 0: h ^= (t >> sr) | (d << sl); h *= m; }
    }
    h ^= h >> 13; h *= m; h ^= h >> 15; return h % max_size;
  } else {
    while(len >= 4) { uint32_t k = *(uint32_t *)data; MIX(h,k,m); data += 4; len -= 4; }
    switch(len) { case 3: h ^= data[2] << 16; [[fallthrough]]; case 2: h ^= data[1] << 8; [[fallthrough]]; case 1: h ^= data[0]; h *= m; }
    h ^= h >> 13; h *= m; h ^= h >> 15; return h % max_size;
  }
}

inline loc_ht& ht_get(loc_ht* thread_ht, cstr_type kmer_key, uint32_t max_size){
  unsigned hash_val = MurmurHashAligned2(kmer_key, max_size);
  while(true){
    if(thread_ht[hash_val].key.length == EMPTY) return thread_ht[hash_val];
    else if(thread_ht[hash_val].key == kmer_key) return thread_ht[hash_val];
    hash_val = (hash_val + 1) % max_size;
  }
}

inline loc_ht_bool& ht_get(loc_ht_bool* thread_ht, cstr_type kmer_key, uint32_t max_size){
  unsigned hash_val = MurmurHashAligned2(kmer_key, max_size);
  while(true){
    if(thread_ht[hash_val].key.length == EMPTY) return thread_ht[hash_val];
    else if(thread_ht[hash_val].key == kmer_key) return thread_ht[hash_val];
    hash_val = (hash_val + 1) % max_size;
  }
}

// Lock-free open-addressing insert/lookup used during count_mers.
// A slot is claimed by atomically installing the key pointer (the lock word), so
// there is no cross-lane busy-wait: colliding sub-group lanes never depend on
// another lane making forward progress, which is required on GPUs that do not
// guarantee independent sub-group forward progress. The slot value (MerFreqs) is
// zeroed by the reset loop before this pass, so the winner need not initialize it,
// avoiding any init race with matching lanes. Every key inserted in a given pass
// has the same length (mer_len), so key equality is a direct content comparison.
inline loc_ht& ht_get_atomic(loc_ht* thread_ht, cstr_type kmer_key, uint32_t max_size){
  unsigned hash_val = MurmurHashAligned2(kmer_key, max_size);
  unsigned orig_hash = hash_val;
  while(true){
    uintptr_t prev = atomicCAS(&thread_ht[hash_val].key.start_ptr,
                               (uintptr_t)0, (uintptr_t)kmer_key.start_ptr);
    if(prev == 0){ // we claimed an empty slot
      thread_ht[hash_val].key.length = kmer_key.length;
      return thread_ht[hash_val];
    }
    // slot occupied: compare stored k-mer content against this key
    const char* stored = (const char*)prev;
    bool eq = true;
    for(int i = 0; i < kmer_key.length; i++){
      if(stored[i] != kmer_key.start_ptr[i]){ eq = false; break; }
    }
    if(eq) return thread_ht[hash_val];
    hash_val = (hash_val + 1) % max_size;
    if(hash_val == orig_hash) return thread_ht[hash_val];
  }
}

inline char walk_mers(loc_ht* thrd_loc_ht, loc_ht_bool* thrd_ht_bool, uint32_t max_ht_size, int& mer_len, cstr_type& mer_walk_temp, cstr_type& longest_walk, cstr_type& walk, const int idx, int max_walk_len){
  char walk_result = 'X';
  for( int nsteps = 0; nsteps < max_walk_len; nsteps++){
    loc_ht_bool &temp_mer_loop = ht_get(thrd_ht_bool, mer_walk_temp, max_walk_len);
    if(temp_mer_loop.key.length == EMPTY){ temp_mer_loop.key = mer_walk_temp; temp_mer_loop.val = true; }
    else{ walk_result = 'R'; break; }
    loc_ht &temp_mer = ht_get(thrd_loc_ht, mer_walk_temp, max_ht_size);
    if(temp_mer.key.length == EMPTY){ walk_result = 'X'; break; }
    char ext = temp_mer.val.ext;
    if(ext == 'F' || ext == 'X'){ walk_result = ext; break; }
    mer_walk_temp.start_ptr = mer_walk_temp.start_ptr + 1;
    mer_walk_temp.start_ptr[mer_walk_temp.length-1] = ext;
    if (ext != 0) walk.length++;
  }
  return walk_result;
}

inline void count_mers(sycl::nd_item<1> &item, sycl::sub_group sg, loc_ht* thrd_loc_ht, char* loc_r_reads, uint32_t max_ht_size, char* loc_r_quals, uint32_t* reads_r_offset, uint32_t& r_rds_cnt,
    uint32_t* rds_count_r_sum, double& loc_ctg_depth, int& mer_len, uint32_t& qual_offset, int64_t& excess_reads, const long int idx){
  const int lane_id = sg.get_local_id()[0];
  cstr_type read;
  cstr_type qual;
  uint32_t running_sum_len = 0;
  for(int i = 0; i < r_rds_cnt; i++){
    read.start_ptr = loc_r_reads + running_sum_len;
    qual.start_ptr = loc_r_quals + running_sum_len;
    if(i == 0){
      if(idx == 0){ read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i]; qual.length = read.length; }
      else{
        if(rds_count_r_sum[idx - 1] == 0){ read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i]; qual.length = read.length; }
        else{ read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i] - reads_r_offset[(rds_count_r_sum[idx - 1] -1)]; qual.length = read.length; }
      }
    } else {
      read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i] - reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + (i-1)];
      qual.length = read.length;
    }
    if (mer_len > read.length) continue;
    int num_mers = read.length - mer_len;
    for( int start = lane_id; start < num_mers; start += sg.get_local_range()[0]){
      cstr_type mer(read.start_ptr + start, mer_len);
      loc_ht &temp_Mer = ht_get_atomic(thrd_loc_ht, mer, max_ht_size);
      int ext_pos = start + mer_len;
      if(ext_pos >= (int) read.length) continue;
      char ext = read.start_ptr[ext_pos];
      if (ext == 'N') continue;
      int qual_diff = qual.start_ptr[ext_pos] - qual_offset;
      if (qual_diff >= LASSM_MIN_QUAL) temp_Mer.val.low_q_exts.inc(ext, 1);
      if (qual_diff >= LASSM_MIN_HI_QUAL) temp_Mer.val.hi_q_exts.inc(ext, 1);
    }
    sycl::group_barrier(sg);
    running_sum_len += read.length;
  }
  sycl::group_barrier(sg);
  for(auto k = (uint32_t)lane_id; k < max_ht_size; k += sg.get_local_range()[0]){
    if(thrd_loc_ht[k].key.length != EMPTY) thrd_loc_ht[k].val.set_ext(loc_ctg_depth);
  }
  sycl::group_barrier(sg);
}

inline void iterative_walks_kernel(sycl::nd_item<1> item,
      uint32_t* cid, uint32_t* ctg_offsets, char* contigs, char* reads_r, char* quals_r, uint32_t* reads_r_offset, uint32_t* rds_count_r_sum,
      double* ctg_depth, loc_ht* global_ht, uint32_t* prefix_ht, loc_ht_bool* global_ht_bool, int kmer_len, uint32_t max_mer_len_off,
      uint32_t *term_counts, int64_t num_walks, int64_t max_walk_len, int64_t sum_ext, int32_t max_read_size, int32_t max_read_count, uint32_t qual_offset,
      char* longest_walks, char* mer_walk_temp, uint32_t* final_walk_lens, int tot_ctgs) {
  auto sg = item.get_sub_group();
  const long int idx = item.get_global_id(0);
  const long int warp_size = sg.get_local_range()[0];
  const long int warp_id_glb = idx / warp_size;
  const long int lane_id = sg.get_local_id()[0];
  if(warp_id_glb < tot_ctgs){
    cstr_type loc_ctg;
    char *loc_r_reads, *loc_r_quals;
    uint32_t r_rds_cnt;
    loc_ht* loc_mer_map;
    uint32_t ht_loc_size;
    loc_ht_bool* loc_bool_map;
    double loc_ctg_depth;
    int64_t excess_reads = 0;
    char* longest_walk_loc;
    char* loc_mer_walk_temp;
    int min_mer_len = LASSM_MIN_KMER_LEN;
    int max_mer_len = LASSM_MAX_KMER_LEN;
    int active = 1;
    if(warp_id_glb == 0){
      loc_ctg.start_ptr = contigs;
      loc_ctg.length = ctg_offsets[warp_id_glb];
      loc_bool_map = global_ht_bool + warp_id_glb * max_walk_len;
      longest_walk_loc = longest_walks + warp_id_glb * max_walk_len;
      loc_mer_walk_temp = mer_walk_temp + warp_id_glb * (max_walk_len + max_mer_len_off);
      r_rds_cnt = rds_count_r_sum[warp_id_glb];
      loc_r_reads = reads_r;
      loc_r_quals = quals_r;
      loc_mer_map = global_ht;
      ht_loc_size = prefix_ht[warp_id_glb];
      loc_ctg_depth = ctg_depth[warp_id_glb];
    }else{
      loc_ctg.start_ptr = contigs + ctg_offsets[warp_id_glb-1];
      loc_ctg.length = ctg_offsets[warp_id_glb] - ctg_offsets[warp_id_glb - 1];
      loc_bool_map = global_ht_bool + warp_id_glb * max_walk_len;
      longest_walk_loc = longest_walks + warp_id_glb * max_walk_len;
      loc_mer_walk_temp = mer_walk_temp + warp_id_glb * (max_walk_len + max_mer_len_off);
      loc_ctg_depth = ctg_depth[warp_id_glb];
      r_rds_cnt = rds_count_r_sum[warp_id_glb] - rds_count_r_sum[warp_id_glb - 1];
      loc_r_reads = (rds_count_r_sum[warp_id_glb - 1] == 0) ? reads_r : reads_r + reads_r_offset[rds_count_r_sum[warp_id_glb - 1] - 1];
      loc_r_quals = (rds_count_r_sum[warp_id_glb - 1] == 0) ? quals_r : quals_r + reads_r_offset[rds_count_r_sum[warp_id_glb - 1] - 1];
      loc_mer_map = global_ht + prefix_ht[warp_id_glb - 1];
      ht_loc_size = prefix_ht[warp_id_glb] - prefix_ht[warp_id_glb - 1];
    }
    uint32_t max_ht_size = ht_loc_size;
    max_mer_len = sycl::min(max_mer_len, loc_ctg.length);
    cstr_type longest_walk_thread(longest_walk_loc,0);
    int shift = 0;
    for(int mer_len = kmer_len; mer_len >= min_mer_len && mer_len <= max_mer_len; mer_len += shift){
      for(uint32_t k = lane_id; k < max_ht_size; k += warp_size) {
        loc_mer_map[k].key.start_ptr = nullptr;
        loc_mer_map[k].key.length = EMPTY;
        loc_mer_map[k].val = {{0,0,0,0}, {0,0,0,0}, 0, 0};
      }
      sycl::group_barrier(sg);
      count_mers(item, sg, loc_mer_map, loc_r_reads, max_ht_size, loc_r_quals, reads_r_offset, r_rds_cnt, rds_count_r_sum, loc_ctg_depth, mer_len, qual_offset, excess_reads, warp_id_glb);
      for(uint32_t k = lane_id; k < max_walk_len; k += warp_size) {
        loc_bool_map[k].key.start_ptr = nullptr;
        loc_bool_map[k].key.length = EMPTY;
      }
      sycl::group_barrier(sg);
      if(lane_id == 0){
        cstr_type ctg_mer(loc_ctg.start_ptr + (loc_ctg.length - mer_len), mer_len);
        cstr_type loc_mer_walk(loc_mer_walk_temp, 0);
        cstr_copy(loc_mer_walk, ctg_mer);
        cstr_type walk(loc_mer_walk.start_ptr + mer_len, 0);
        char walk_res = walk_mers(loc_mer_map, loc_bool_map, max_ht_size, mer_len, loc_mer_walk, longest_walk_thread, walk, warp_id_glb, max_walk_len);
        if (walk.length > longest_walk_thread.length) cstr_copy(longest_walk_thread, walk);
        if (walk_res == 'X') { if (shift == LASSM_SHIFT_SIZE) active = 0; shift = -LASSM_SHIFT_SIZE; }
        else {
          if (shift == -LASSM_SHIFT_SIZE || mer_len > loc_ctg.length) active = 0;
          shift = LASSM_SHIFT_SIZE;
        }
      }
      active = bcast_warp(sg, active);
      if(active == 0) break;
      shift = bcast_warp(sg, shift);
      sycl::group_barrier(sg);
    }
    if(lane_id == 0) final_walk_lens[warp_id_glb] = (longest_walk_thread.length > 0) ? longest_walk_thread.length : 0;
  }
}

inline void launch_iterative_walks_kernel(sycl::queue &q, unsigned blocks, unsigned threads,
      uint32_t* cid, uint32_t* ctg_offsets, char* contigs, char* reads_r, char* quals_r, uint32_t* reads_r_offset, uint32_t* rds_count_r_sum,
      double* ctg_depth, loc_ht* global_ht, uint32_t* prefix_ht, loc_ht_bool* global_ht_bool, int kmer_len, uint32_t max_mer_len_off,
      uint32_t *term_counts, int64_t num_walks, int64_t max_walk_len, int64_t sum_ext, int32_t max_read_size, int32_t max_read_count, uint32_t qual_offset,
      char* longest_walks, char* mer_walk_temp, uint32_t* final_walk_lens, int tot_ctgs) {
  size_t local = threads;
  size_t global = blocks * threads;
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) {
      iterative_walks_kernel(item, cid, ctg_offsets, contigs, reads_r, quals_r, reads_r_offset, rds_count_r_sum,
          ctg_depth, global_ht, prefix_ht, global_ht_bool, kmer_len, max_mer_len_off, term_counts, num_walks, max_walk_len,
          sum_ext, max_read_size, max_read_count, qual_offset, longest_walks, mer_walk_temp, final_walk_lens, tot_ctgs);
    });
  }).wait();
}
