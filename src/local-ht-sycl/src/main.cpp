#include <unordered_map>
#include <typeinfo>
#include <random>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include "helper.hpp"
#include "kernel.hpp"
#include <memory>
#include <sycl/sycl.hpp>


size_t get_device_mem(){
  size_t free_mem, total_mem;
  auto dev = sycl::queue{}.get_device();
  total_mem = dev.get_info<sycl::info::device::global_mem_size>();
  free_mem = total_mem;
  std::cout << "SYCL device has reported memory (Mbytes):=" << (double)free_mem/(1024*1024) << std::endl;
  return free_mem;
}

struct accum_data{
  std::vector<uint32_t> ht_sizes;
  std::vector<uint32_t> l_reads_count;
  std::vector<uint32_t> r_reads_count;
  std::vector<uint32_t> ctg_sizes;
};


void call_kernel(std::vector<CtgWithReads>& data_in, uint32_t max_ctg_size, uint32_t max_read_size, uint32_t max_r_count, uint32_t max_l_count, int mer_len,int max_reads_count, accum_data& sizes_outliers, std::ofstream& out_file_g);

// sample cmd line: ./build/ht_loc ../locassm_data/localassm_extend_7-21.dat <kmer_size> ./out_file 
int main (int argc, char* argv[]){
  if(argc != 4){
    std::cout << "argc:"<<argc<<std::endl;
    std::cout<< "Usage:\n";
    std::cout<< "./build/ht_loc ../locassm_data/localassm_extend_7-21.dat 21 ./out_file.dat"<<std::endl;
    return 0;
  }

  std::string in_file = argv[1];
  int max_mer_len = std::stoi(argv[2]);
  std::ofstream ofile(argv[3]);
  std::vector<CtgWithReads> data_in;
  uint32_t max_ctg_size, total_r_reads, total_l_reads, max_read_size, max_r_count, max_l_count;
  read_locassm_data(&data_in, in_file, max_ctg_size, total_r_reads, total_l_reads, max_read_size, max_r_count, max_l_count);
  timer overall_time;
  overall_time.timer_start();

  std::vector<CtgWithReads> zero_slice, mid_slice, midsup_slice, outlier_slice;
  uint32_t mid_l_max = 0, mid_r_max = 0, outlier_l_max = 0, outlier_r_max = 0, mid_max_contig_sz = 0;
  uint32_t outliers_max_contig_sz = 0, mids_tot_r_reads = 0, mids_tot_l_reads = 0, outliers_tot_r_reads = 0;
  uint32_t outliers_tot_l_reads = 0;
  accum_data sizes_mid, sizes_outliers;

  for(int i = 0; i < data_in.size(); i++){
    CtgWithReads temp_in = data_in[i];
    if(temp_in.max_reads == 0){
      zero_slice.push_back(temp_in);
    }else if(temp_in.max_reads > 0 && temp_in.max_reads < 10){
      mid_slice.push_back(temp_in);
      uint32_t temp_ht_size = temp_in.max_reads * max_read_size;
      sizes_mid.ht_sizes.push_back(temp_ht_size);
      sizes_mid.ctg_sizes.push_back(temp_in.seq.size());
      sizes_mid.l_reads_count.push_back(temp_in.reads_left.size());
      sizes_mid.r_reads_count.push_back(temp_in.reads_right.size());
      mids_tot_r_reads += temp_in.reads_right.size();
      mids_tot_l_reads += temp_in.reads_left.size();
      if(mid_l_max < temp_in.reads_left.size())
        mid_l_max = temp_in.reads_left.size();
      if(mid_r_max < temp_in.reads_right.size())
        mid_r_max = temp_in.reads_right.size();
      if(mid_max_contig_sz < temp_in.seq.size())
        mid_max_contig_sz = temp_in.seq.size();
    }else{
      outlier_slice.push_back(temp_in);
      uint32_t temp_ht_size = temp_in.max_reads * max_read_size;
      sizes_outliers.ht_sizes.push_back(temp_ht_size);
      sizes_outliers.ctg_sizes.push_back(temp_in.seq.size());
      sizes_outliers.l_reads_count.push_back(temp_in.reads_left.size());
      sizes_outliers.r_reads_count.push_back(temp_in.reads_right.size());
      outliers_tot_r_reads += temp_in.reads_right.size();
      outliers_tot_l_reads += temp_in.reads_left.size();
      if(outlier_l_max < temp_in.reads_left.size())
        outlier_l_max = temp_in.reads_left.size();
      if(outlier_r_max < temp_in.reads_right.size())
        outlier_r_max = temp_in.reads_right.size();
      if(outliers_max_contig_sz < temp_in.seq.size())
        outliers_max_contig_sz = temp_in.seq.size();
    }
  }

  print_vals("zeroes, count:", zero_slice.size());
  timer file_time;
  file_time.timer_start();
  for(int i = 0; i < zero_slice.size(); i++){
    ofile << zero_slice[i].cid<<" "<<zero_slice[i].seq<<std::endl;

  }
  file_time.timer_end();
  print_vals("zeroes file write time:",file_time.get_total_time());
  zero_slice = std::vector<CtgWithReads>();
  data_in = std::vector<CtgWithReads>();
  print_vals("mids calling",  "mids count:", mid_slice.size());

  int max_reads_count = 10;
  call_kernel(mid_slice, mid_max_contig_sz, max_read_size, mid_r_max, mid_l_max, max_mer_len,max_reads_count, sizes_mid, ofile);

  print_vals("outliers calling", "outliers count:", outlier_slice.size());
  call_kernel(outlier_slice, outliers_max_contig_sz, max_read_size, outlier_r_max, outlier_l_max, max_mer_len, max_reads_count, sizes_outliers, ofile);
  overall_time.timer_end();

  print_vals("Total Time including file write:", overall_time.get_total_time());

  ofile.flush();
  ofile.close();

  return 0;
}


void call_kernel(std::vector<CtgWithReads>& data_in, uint32_t max_ctg_size, uint32_t max_read_size, uint32_t max_r_count, uint32_t max_l_count, int mer_len, int max_reads_count, accum_data& sizes_vecs, std::ofstream& out_file_g)
{
  sycl::queue q{sycl::default_selector_v};
  std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  unsigned warp_sz = 32;
  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (!sg_sizes.empty())
    warp_sz = *std::max_element(sg_sizes.begin(), sg_sizes.end());

  int max_mer_len = LASSM_MAX_KMER_LEN;//mer_len;

  unsigned tot_extensions = data_in.size();
  uint32_t max_read_count = max_r_count>max_l_count ? max_r_count : max_l_count;
  int insert_avg = 121;
  int insert_stddev = 246;
  int max_walk_len = insert_avg + 2 * insert_stddev;
  uint32_t ht_tot_size = std::accumulate(sizes_vecs.ht_sizes.begin(), sizes_vecs.ht_sizes.end(), 0);
  uint32_t total_r_reads = std::accumulate(sizes_vecs.r_reads_count.begin(), sizes_vecs.r_reads_count.end(), 0);
  uint32_t total_l_reads = std::accumulate(sizes_vecs.l_reads_count.begin(), sizes_vecs.l_reads_count.end(), 0);
  uint32_t total_ctg_len = std::accumulate(sizes_vecs.ctg_sizes.begin(), sizes_vecs.ctg_sizes.end(), 0);
  print_vals("new HT size:", ht_tot_size*sizeof(loc_ht));
  size_t gpu_mem_req = sizeof(int32_t) * tot_extensions * 6 + sizeof(int32_t) * total_l_reads
    + sizeof(int32_t) * total_r_reads + sizeof(char) * total_ctg_len
    + sizeof(char) * total_l_reads * max_read_size*2 + sizeof(char) * total_r_reads * max_read_size*2 // for quals included
    + sizeof(double) * tot_extensions + sizeof(char) * total_r_reads * max_read_size 
    + sizeof(char) * total_l_reads * max_read_size + sizeof(int64_t)*3
    + sizeof(loc_ht)*ht_tot_size // changed to try the new method
    + sizeof(char)*tot_extensions * max_walk_len
    + (max_mer_len + max_walk_len) * sizeof(char) * tot_extensions
    + sizeof(loc_ht_bool) * tot_extensions * max_walk_len;


  print_vals("Total GPU mem required (GBs):", (double)gpu_mem_req/(1024*1024*1024));                     
  size_t gpu_mem_avail = get_device_mem();
  float factor = 0.90;
  print_vals("GPU Mem using (MB):",((double)gpu_mem_avail*factor)/(1024*1024)); 
  int iterations = ceil(((double)gpu_mem_req)/((double)gpu_mem_avail*factor)); // 0.8 is to buffer for the extra mem that is used when allocating once and using again
  print_vals("Iterations:", iterations);
  unsigned slice_size = tot_extensions/iterations;
  unsigned remaining = tot_extensions % iterations;
  std::vector<uint32_t> max_ht_sizes;
  //to get the largest ht size for any iteration and allocate GPU memory for that (once)
  uint32_t max_ht = 0, max_r_rds_its = 0, max_l_rds_its = 0, max_ctg_len_it = 0, test_sum = 0;
  for(int i = 0; i < iterations; i++){
    uint32_t temp_max_ht = 0, temp_max_r_rds = 0, temp_max_l_rds = 0, temp_max_ctg_len = 0;
    if(i < iterations -1 ){
      temp_max_ht = std::accumulate(sizes_vecs.ht_sizes.begin() + i*slice_size, sizes_vecs.ht_sizes.begin()+(i+1)*slice_size, 0 );
      temp_max_r_rds = std::accumulate(sizes_vecs.r_reads_count.begin() + i*slice_size, sizes_vecs.r_reads_count.begin()+(i+1)*slice_size, 0 );
      temp_max_l_rds = std::accumulate(sizes_vecs.l_reads_count.begin() + i*slice_size, sizes_vecs.l_reads_count.begin()+(i+1)*slice_size, 0 );
      temp_max_ctg_len = std::accumulate(sizes_vecs.ctg_sizes.begin() + i*slice_size, sizes_vecs.ctg_sizes.begin()+(i+1)*slice_size, 0 );
    }
    else{
      temp_max_ht = std::accumulate(sizes_vecs.ht_sizes.begin() + i*slice_size, sizes_vecs.ht_sizes.begin()+((i+1)*slice_size) + remaining, 0 );
      temp_max_r_rds = std::accumulate(sizes_vecs.r_reads_count.begin() + i*slice_size, sizes_vecs.r_reads_count.begin()+((i+1)*slice_size) + remaining, 0 );
      temp_max_l_rds = std::accumulate(sizes_vecs.l_reads_count.begin() + i*slice_size, sizes_vecs.l_reads_count.begin()+((i+1)*slice_size) + remaining, 0 );
      temp_max_ctg_len = std::accumulate(sizes_vecs.ctg_sizes.begin() + i*slice_size, sizes_vecs.ctg_sizes.begin()+((i+1)*slice_size) + remaining, 0 );
    }
    if(temp_max_ht > max_ht)
      max_ht = temp_max_ht;
    if(temp_max_r_rds > max_r_rds_its)
      max_r_rds_its = temp_max_r_rds;
    if(temp_max_l_rds > max_l_rds_its)
      max_l_rds_its = temp_max_l_rds; 
    if(temp_max_ctg_len > max_ctg_len_it)
      max_ctg_len_it = temp_max_ctg_len;
    test_sum += temp_max_ht;
  }

  print_vals("test_sum:", test_sum*sizeof(loc_ht));
  print_vals("slice size regular:", slice_size);
  slice_size = slice_size + remaining; // this is the largest slice size, mostly the last iteration handles the leftovers
  //allocating maximum possible memory for a single iteration
  print_vals("slice size maximum:", slice_size);
  print_vals("max ctg size:", max_ctg_size, "slice size:",slice_size, "product:", max_ctg_size*slice_size);
  size_t cpu_mem_est = sizeof(int32_t)*slice_size * 4
    + sizeof(char) * max_ctg_size * slice_size * 2
    + sizeof(double) * slice_size
    + sizeof(char) * max_l_count * max_read_size * slice_size
    + sizeof(char) * max_r_count * max_read_size * slice_size
    + sizeof(int32_t) * max_l_count* slice_size
    + sizeof(int32_t) * max_r_count* slice_size
    + sizeof(char) * slice_size * max_walk_len * iterations * 2
    + sizeof(int) * slice_size * iterations * 2;
  print_vals("cpu_mem:", cpu_mem_est);
  timer mem_timer;
  double cpu_mem_aloc_time = 0, gpu_mem_aloc_time = 0, cpu_mem_dealoc_time = 0, gpu_mem_dealoc_time = 0;

  mem_timer.timer_start();
  auto ctg_seqs_h            = std::make_unique<char[]>(max_ctg_size * slice_size);
  auto cid_h                 = std::make_unique<uint32_t[]>(slice_size);
  auto ctgs_seqs_rc_h        = std::make_unique<char[]>(max_ctg_size * slice_size);
  auto ctg_seq_offsets_h     = std::make_unique<uint32_t[]>(slice_size);
  auto depth_h               = std::make_unique<double[]>(slice_size);
  auto reads_left_h          = std::make_unique<char[]>(max_l_count * max_read_size * slice_size);
  auto reads_right_h         = std::make_unique<char[]>(max_r_count * max_read_size * slice_size);
  auto quals_right_h         = std::make_unique<char[]>(max_r_count * max_read_size * slice_size);
  auto quals_left_h          = std::make_unique<char[]>(max_l_count * max_read_size * slice_size);
  auto reads_l_offset_h      = std::make_unique<uint32_t[]>(max_l_count * slice_size);
  auto reads_r_offset_h      = std::make_unique<uint32_t[]>(max_r_count * slice_size);
  auto rds_l_cnt_offset_h    = std::make_unique<uint32_t[]>(slice_size);
  auto rds_r_cnt_offset_h    = std::make_unique<uint32_t[]>(slice_size);
  auto term_counts_h         = std::make_unique<uint32_t[]>(3);
  auto longest_walks_r_h     = std::make_unique<char[]>(slice_size * max_walk_len * iterations);
  auto longest_walks_l_h     = std::make_unique<char[]>(slice_size * max_walk_len * iterations);
  auto final_walk_lens_r_h   = std::make_unique<uint32_t[]>(slice_size * iterations);
  auto final_walk_lens_l_h   = std::make_unique<uint32_t[]>(slice_size * iterations);
  auto prefix_ht_size_h      = std::make_unique<uint32_t[]>(slice_size);
  mem_timer.timer_end();
  cpu_mem_aloc_time += mem_timer.get_total_time();
  gpu_mem_req = sizeof(int32_t) * slice_size * 6 + sizeof(int32_t) * 3
    + sizeof(int32_t) * max_l_rds_its
    + sizeof(int32_t) * max_r_rds_its
    + sizeof(char) * max_ctg_len_it
    + sizeof(char) * max_l_rds_its * max_read_size * 2
    + sizeof(char) * max_r_rds_its * max_read_size  * 2
    + sizeof(double) * slice_size 
    + sizeof(loc_ht) * max_ht
    + sizeof(char) * slice_size * max_walk_len
    + (max_mer_len + max_walk_len) * sizeof(char) * slice_size
    + sizeof(loc_ht_bool) * slice_size * max_walk_len;

  print_vals("Device Mem requesting per slice (MB):", (double)gpu_mem_req/ (1024*1024));

  print_vals("**old lochash size:",sizeof(loc_ht)*(max_read_size*max_read_count)*slice_size, "new local hash:",sizeof(loc_ht) * max_ht);
  print_vals("**boolhash size:",sizeof(loc_ht_bool) * slice_size * max_walk_len);
  print_vals("**max_read_count:", max_read_count);
  print_vals("max read l count:", max_l_count);
  print_vals("max read r count:", max_r_count);
  print_vals("size of loc_host:", sizeof(loc_ht));

  uint32_t *cid_d, *ctg_seq_offsets_d, *reads_l_offset_d, *reads_r_offset_d; 
  uint32_t *rds_l_cnt_offset_d, *rds_r_cnt_offset_d, *prefix_ht_size_d;
  char *ctg_seqs_d, *reads_left_d, *reads_right_d, *quals_left_d, *quals_right_d;
  char *longest_walks_d, *mer_walk_temp_d;
  double *depth_d;
  uint32_t *term_counts_d;
  loc_ht *d_ht;
  loc_ht_bool *d_ht_bool;
  uint32_t* final_walk_lens_d;
  mem_timer.timer_start();
  //allocate GPU  memory
  prefix_ht_size_d = sycl::malloc_device<uint32_t>(slice_size, q);
  cid_d = sycl::malloc_device<uint32_t>(slice_size, q);
  ctg_seq_offsets_d = sycl::malloc_device<uint32_t>(slice_size, q);
  reads_l_offset_d = sycl::malloc_device<uint32_t>(max_l_rds_its, q);
  reads_r_offset_d = sycl::malloc_device<uint32_t>(max_r_rds_its, q);
  rds_l_cnt_offset_d = sycl::malloc_device<uint32_t>(slice_size, q);
  rds_r_cnt_offset_d = sycl::malloc_device<uint32_t>(slice_size, q);
  ctg_seqs_d = sycl::malloc_device<char>(max_ctg_len_it, q);
  reads_left_d = sycl::malloc_device<char>(max_read_size * max_l_rds_its, q);
  reads_right_d = sycl::malloc_device<char>(max_read_size * max_r_rds_its, q);
  depth_d = sycl::malloc_device<double>(slice_size, q);
  quals_right_d = sycl::malloc_device<char>(max_read_size * max_r_rds_its, q);
  quals_left_d = sycl::malloc_device<char>(max_read_size * max_l_rds_its, q);
  term_counts_d = sycl::malloc_device<uint32_t>(3, q);
  // if we separate out kernels for right and left walks then we can use r_count/l_count separately but for now use the max of two
  // also subtract the appropriate kmer length from max_read_size to reduce memory footprint of global ht_loc.
  // one local hashtable for each thread, so total hash_tables equal to vec_size i.e. total contigs
  d_ht = sycl::malloc_device<loc_ht>(max_ht, q);
  longest_walks_d = sycl::malloc_device<char>(slice_size * max_walk_len, q);
  mer_walk_temp_d = sycl::malloc_device<char>((max_mer_len + max_walk_len) * slice_size, q);
  d_ht_bool = sycl::malloc_device<loc_ht_bool>(slice_size * max_walk_len, q);
  final_walk_lens_d = sycl::malloc_device<uint32_t>(slice_size, q);
  mem_timer.timer_end();
  gpu_mem_aloc_time += mem_timer.get_total_time();



  //start a loop here which takes a slice of data_in
  //performs data packing on that slice on cpu memory
  //and then moves that data to allocated GPU memory
  //calls the kernels, revcomps, copy backs walks and
  timer loop_time;
  loop_time.timer_start();
  double data_mv_tim = 0;
  double packing_tim = 0;
  double kernel_tim = 0;
  slice_size = tot_extensions/iterations;
  for(int slice = 0; slice < iterations; slice++){
    print_vals("Done(%):", ((double)slice/iterations)*100);
    uint32_t left_over;
    if(iterations - 1 == slice)
      left_over = tot_extensions % iterations;
    else
      left_over = 0;
    std::vector<CtgWithReads> slice_data (&data_in[slice*slice_size], &data_in[(slice + 1)*slice_size + left_over]);
    uint32_t vec_size = slice_data.size();
    print_vals("current_slice vec_size:", vec_size);
    print_vals("current_slice slice_size:", slice_size);

    print_vals("Starting Data Packing");
    uint32_t ctgs_offset_sum = 0;
    uint32_t prefix_ht_sum = 0;
    uint32_t reads_r_offset_sum = 0;
    uint32_t reads_l_offset_sum = 0;
    int read_l_index = 0, read_r_index = 0;
    timer tim_temp;
    tim_temp.timer_start();
    for(int i = 0; i < slice_data.size(); i++){
      CtgWithReads temp_data = slice_data[i];
      cid_h[i] = temp_data.cid;
      depth_h[i] = temp_data.depth;
      //convert string to c-string
      char *ctgs_ptr = ctg_seqs_h.get() + ctgs_offset_sum;
      memcpy(ctgs_ptr, temp_data.seq.c_str(), temp_data.seq.size());
      ctgs_offset_sum += temp_data.seq.size();
      ctg_seq_offsets_h[i] = ctgs_offset_sum;
      prefix_ht_sum += temp_data.max_reads * max_read_size;
      prefix_ht_size_h[i] = prefix_ht_sum;

      for(int j = 0; j < temp_data.reads_left.size(); j++){
        char *reads_l_ptr = reads_left_h.get() + reads_l_offset_sum;
        char *quals_l_ptr = quals_left_h.get() + reads_l_offset_sum;
        memcpy(reads_l_ptr, temp_data.reads_left[j].seq.c_str(), temp_data.reads_left[j].seq.size());
        //quals offsets will be same as reads offset because quals and reads have same length
        memcpy(quals_l_ptr, temp_data.reads_left[j].quals.c_str(), temp_data.reads_left[j].quals.size());
        reads_l_offset_sum += temp_data.reads_left[j].seq.size();
        reads_l_offset_h[read_l_index] = reads_l_offset_sum;
        read_l_index++;
      }
      rds_l_cnt_offset_h[i] = read_l_index; // running sum of left reads count

      for(int j = 0; j < temp_data.reads_right.size(); j++){
        char *reads_r_ptr = reads_right_h.get() + reads_r_offset_sum;
        char *quals_r_ptr = quals_right_h.get() + reads_r_offset_sum;
        memcpy(reads_r_ptr, temp_data.reads_right[j].seq.c_str(), temp_data.reads_right[j].seq.size());
        //quals offsets will be same as reads offset because quals and reads have same length
        memcpy(quals_r_ptr, temp_data.reads_right[j].quals.c_str(), temp_data.reads_right[j].quals.size());
        reads_r_offset_sum += temp_data.reads_right[j].seq.size();
        reads_r_offset_h[read_r_index] = reads_r_offset_sum;
        read_r_index++;
      }
      rds_r_cnt_offset_h[i] = read_r_index; // running sum of right reads count
    }// data packing for loop ends
    tim_temp.timer_end();
    packing_tim += tim_temp.get_total_time();

    int total_r_reads_slice = read_r_index;
    int total_l_reads_slice = read_l_index;

    for(int i = 0; i < 3; i++){
      term_counts_h[i] = 0;
    }

    tim_temp.timer_start();
    print_vals("Host to Device Transfer...");
    q.memcpy(prefix_ht_size_d, prefix_ht_size_h.get(), sizeof(uint32_t) * (vec_size));
    q.memcpy(cid_d, cid_h.get(), sizeof(uint32_t) * (vec_size));
    q.memcpy(ctg_seq_offsets_d, ctg_seq_offsets_h.get(), sizeof(uint32_t) * (vec_size));
    q.memcpy(reads_l_offset_d, reads_l_offset_h.get(), sizeof(uint32_t) * (total_l_reads_slice));
    q.memcpy(reads_r_offset_d, reads_r_offset_h.get(), sizeof(uint32_t) * (total_r_reads_slice));
    q.memcpy(rds_l_cnt_offset_d, rds_l_cnt_offset_h.get(), sizeof(uint32_t) * (vec_size));
    q.memcpy(rds_r_cnt_offset_d, rds_r_cnt_offset_h.get(), sizeof(uint32_t) * (vec_size));
    q.memcpy(ctg_seqs_d, ctg_seqs_h.get(), sizeof(char) * (ctgs_offset_sum));
    q.memcpy(reads_left_d, reads_left_h.get(), sizeof(char) * (reads_l_offset_sum));
    q.memcpy(reads_right_d, reads_right_h.get(), sizeof(char) * (reads_r_offset_sum));
    q.memcpy(depth_d, depth_h.get(), sizeof(double) * (vec_size));
    q.memcpy(quals_right_d, quals_right_h.get(), sizeof(char) * (reads_r_offset_sum));
    q.memcpy(quals_left_d, quals_left_h.get(), sizeof(char) * (reads_l_offset_sum));
    q.memcpy(term_counts_d, term_counts_h.get(), sizeof(uint32_t) * (3));
    q.wait();
    tim_temp.timer_end();
    data_mv_tim += tim_temp.get_total_time();
    //call kernel here, one thread per contig
    unsigned total_threads = vec_size*warp_sz;// we need one warp (warp_sz threads) per extension, vec_size = extensions
    unsigned thread_per_blk = 512;
    unsigned blocks = (total_threads + thread_per_blk - 1)/thread_per_blk;


    print_vals("Calling Kernel with blocks:", blocks, "Threads:", thread_per_blk);
    int64_t sum_ext=0, num_walks=0;
    uint32_t qual_offset = 33;
    tim_temp.timer_start();
    launch_iterative_walks_kernel(q, blocks, thread_per_blk, cid_d, ctg_seq_offsets_d, ctg_seqs_d, reads_right_d, quals_right_d, reads_r_offset_d, rds_r_cnt_offset_d,
        depth_d, d_ht, prefix_ht_size_d, d_ht_bool, mer_len, max_mer_len, term_counts_d, num_walks, max_walk_len, sum_ext, max_read_size, max_read_count, qual_offset, longest_walks_d, mer_walk_temp_d, final_walk_lens_d, vec_size);
    tim_temp.timer_end();
    kernel_tim += tim_temp.get_total_time();

    //perform revcomp of contig sequences and launch kernel with left reads, 
    print_vals("revcomp-ing the contigs for next kernel");
    // timer rev_comp_;
    // rev_comp_.timer_start();
    for(int j = 0; j < vec_size; j++){
      int size_lst;
      char* curr_seq;
      char* curr_seq_rc;
      if(j == 0){
        size_lst = ctg_seq_offsets_h[j];
        curr_seq = ctg_seqs_h.get();
        curr_seq_rc = ctgs_seqs_rc_h.get();
      }
      else{
        size_lst = ctg_seq_offsets_h[j] - ctg_seq_offsets_h[j-1];
        curr_seq = ctg_seqs_h.get() + ctg_seq_offsets_h[j - 1];
        curr_seq_rc = ctgs_seqs_rc_h.get() + ctg_seq_offsets_h[j - 1];
      }
      revcomp(curr_seq, curr_seq_rc, size_lst);
#ifdef DEBUG_PRINT_CPU   
      print_vals("orig seq:");
      for(int h = 0; h < size_lst; h++)
        std::cout<<curr_seq[h];
      std::cout << std::endl;
      print_vals("recvomp seq:");
      for(int h = 0; h < size_lst; h++)
        std::cout<<curr_seq_rc[h];
      std::cout << std::endl;    
#endif   

    }
    tim_temp.timer_start();
    q.memcpy(longest_walks_r_h.get() + slice * max_walk_len * slice_size, longest_walks_d, sizeof(char) * (vec_size * max_walk_len));
    q.memcpy(final_walk_lens_r_h.get() + slice * slice_size, final_walk_lens_d, sizeof(int32_t) * (vec_size));

    //cpying rev comped ctgs to device on same memory as previous ctgs
    q.memcpy(ctg_seqs_d, ctgs_seqs_rc_h.get(), sizeof(char) * (ctgs_offset_sum));
    q.wait();
    tim_temp.timer_end();
    data_mv_tim += tim_temp.get_total_time();

    tim_temp.timer_start();
    launch_iterative_walks_kernel(q, blocks, thread_per_blk, cid_d, ctg_seq_offsets_d, ctg_seqs_d, reads_left_d, quals_left_d, reads_l_offset_d, rds_l_cnt_offset_d,
        depth_d, d_ht, prefix_ht_size_d, d_ht_bool, mer_len, max_mer_len, term_counts_d, num_walks, max_walk_len, sum_ext, max_read_size, max_read_count, qual_offset, longest_walks_d, mer_walk_temp_d, final_walk_lens_d, vec_size);
    tim_temp.timer_end();
    kernel_tim += tim_temp.get_total_time();

    print_vals("Device to Host Transfer...", "Copying back left walks");
    tim_temp.timer_start();
    q.memcpy(longest_walks_l_h.get() + slice * max_walk_len * slice_size , longest_walks_d, sizeof(char) * (vec_size * max_walk_len)); // copy back left walks
    q.memcpy(final_walk_lens_l_h.get() + slice * slice_size , final_walk_lens_d, sizeof(int32_t) * (vec_size)); 
    q.wait();
    tim_temp.timer_end();
    data_mv_tim += tim_temp.get_total_time();
  }// the big for loop over all slices ends here
  loop_time.timer_end();
  print_vals("Total Loop Time:", loop_time.get_total_time());
  print_vals("Total Data Move Time:", data_mv_tim);
  print_vals("Total Packing Time:", packing_tim);
  std::cerr << "Total Kernel Time (s):" << kernel_tim << std::endl;

  //once all the alignments are on cpu, then go through them and stitch them with contigs in front and back.

  int loc_left_over = tot_extensions % iterations;
  for(int j = 0; j < iterations; j++){
    int loc_size = (j == iterations - 1) ? slice_size + loc_left_over : slice_size;

    for(int i = 0; i< loc_size; i++){
      if(final_walk_lens_l_h[j*slice_size + i] != 0){
        std::string left(longest_walks_l_h.get() + j*slice_size*max_walk_len + max_walk_len*i,final_walk_lens_l_h[j*slice_size + i]);
        std::string left_rc = revcomp(left);
        //print_vals("cid:",data_in[j*slice_size + i].cid, "walk:",left, "length:",final_walk_lens_l_h[j*slice_size + i]);
        data_in[j*slice_size + i].seq.insert(0,left_rc);  
      }
      if(final_walk_lens_r_h[j*slice_size + i] != 0){
        std::string right(longest_walks_r_h.get() + j*slice_size*max_walk_len + max_walk_len*i,final_walk_lens_r_h[j*slice_size + i]);
        data_in[j*slice_size + i].seq += right;
      }
      out_file_g << data_in[j*slice_size + i].cid<<" "<<data_in[j*slice_size + i].seq<<std::endl;
    }
  }


  mem_timer.timer_start();  
  sycl::free(term_counts_d, q);
  sycl::free(cid_d, q);
  sycl::free(ctg_seq_offsets_d, q);
  sycl::free(reads_l_offset_d, q);
  sycl::free(reads_r_offset_d, q);
  sycl::free(rds_l_cnt_offset_d, q);
  sycl::free(rds_r_cnt_offset_d, q);
  sycl::free(ctg_seqs_d, q);
  sycl::free(reads_left_d, q);
  sycl::free(reads_right_d, q);
  sycl::free(depth_d, q);
  sycl::free(quals_right_d, q);
  sycl::free(quals_left_d, q);
  sycl::free(d_ht, q); 
  sycl::free(longest_walks_d, q);
  sycl::free(mer_walk_temp_d, q);
  sycl::free(d_ht_bool, q);
  sycl::free(final_walk_lens_d, q);
  mem_timer.timer_end();
  gpu_mem_dealoc_time += mem_timer.get_total_time();

  mem_timer.timer_start();
  mem_timer.timer_end();
  cpu_mem_dealoc_time += mem_timer.get_total_time();
  print_vals("gpu_aloc_time:", gpu_mem_aloc_time, "gpu_dealoc_time:",gpu_mem_dealoc_time, "cpu_aloc_time:", cpu_mem_aloc_time, "cpu_dealoc_time:", cpu_mem_dealoc_time);

}
