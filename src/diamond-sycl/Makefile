#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_70
HIP       = no
HIP_ARCH  ?= gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

c_src= \
  src/blast/sm_blosum45.c \
  src/blast/sm_blosum50.c \
  src/blast/sm_blosum62.c \
  src/blast/sm_blosum80.c \
  src/blast/sm_blosum90.c \
  src/blast/sm_pam30.c \
  src/blast/sm_pam70.c \
  src/blast/sm_pam250.c

cpp_src = \
  src/run/main.cpp \
  src/basic/config.cpp \
  src/util/tinythread.cpp \
  src/util/compressed_stream.cpp \
  src/basic/score_matrix.cpp \
  src/blast/blast_filter.cpp \
  src/blast/blast_seg.cpp \
  src/data/queries.cpp \
  src/data/reference.cpp \
  src/data/seed_histogram.cpp \
  src/output/daa_record.cpp \
  src/search/search.cpp \
  src/util/command_line_parser.cpp \
  src/util/seq_file_format.cpp \
  src/util/util.cpp  \
  src/util/Timer.cpp \
  src/basic/basic.cpp \
  src/dp/floating_sw.cpp \
  src/basic/hssp.cpp \
  src/dp/ungapped_align.cpp \
  src/run/tools.cpp \
  src/dp/greedy_align.cpp \
  src/run/benchmark.cpp \
  src/search/stage2.cpp \
  src/output/output_format.cpp \
  src/output/join_blocks.cpp \
  src/data/frequent_seeds.cpp \
  src/align/query_mapper.cpp \
  src/align/align_target.cpp \
  src/output/blast_tab_format.cpp \
  src/dp/padded_banded_sw.cpp \
  src/dp/needleman_wunsch.cpp \
  src/output/blast_pairwise_format.cpp \
  src/extra/roc.cpp \
  src/dp/comp_based_stats.cpp \
  src/extra/model_sim.cpp \
  src/run/double_indexed.cpp \
  src/search/collision.cpp \
  src/output/sam_format.cpp \
  src/align/align.cpp \
  src/search/setup.cpp \
  src/extra/opt.cpp \
  src/dp/diag_scores.cpp \
  src/data/taxonomy.cpp \
  src/lib/tantan/tantan.cpp \
  src/lib/tantan/tantale.cpp \
  src/basic/masking.cpp \
  src/dp/swipe.cpp \
  src/dp/banded_sw.cpp \
  src/data/sorted_list.cpp \
  src/data/seed_set.cpp \
  src/util/binary_file.cpp \
  src/output/taxon_format.cpp \
  src/output/view.cpp \
  src/output/output_sink.cpp \
  src/output/target_culling.cpp

obj = $(c_src:.c=.o) $(cpp_src:.cpp=.o) 

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = -lz -lpthread

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH) 
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.m8

test: $(program)
	./$(program) blastx -q short.fastq.gz -d ARDB -o short_cur.m8 --tmpdir /dev/shm -p1

run: $(program)
	$(LAUNCHER) ./$(program) blastx -q long.fastq.gz -d ARDB -o long_cur.m8 --tmpdir /dev/shm -p1

