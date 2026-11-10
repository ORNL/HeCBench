#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = src/main.cu

# Header-only dependencies previously pulled in by the upstream CMake project
CXXOPTS_INC   = external/cxxopts/include
FMT_INC       = external/fmt/include
ZIPF_INC      = external/zipf
PROGRESS_INC  = external/progressbar/include/progresscpp

INCLUDES = -Iinclude -I$(CXXOPTS_INC) -I$(FMT_INC)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -DFMT_HEADER_ONLY $(INCLUDES) -arch=$(ARCH)

LDFLAGS = -lcublas

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

.PHONY: all clean run deps

all: $(program)

deps: external/.stamp

external/.stamp:
	mkdir -p external
	test -d external/cxxopts || git clone --depth 1 --branch v3.3.1 https://github.com/jarro2783/cxxopts.git external/cxxopts
	test -d external/fmt || git clone --depth 1 --branch 7.1.2 https://github.com/fmtlib/fmt.git external/fmt
	test -d external/zipf || git clone https://github.com/llersch/cpp_random_distributions.git external/zipf
	git -C external/zipf checkout 6ae38ecf5dbeb1150b80ecf051cb43cb88ba697a
	test -d external/progressbar || git clone https://github.com/prakhar1989/progress-cpp.git external/progressbar
	git -C external/progressbar checkout 7bfba0d22d19c41323aa35541618b6ebec9d737c
	touch $@

$(program): $(source) | external/.stamp
	$(CC) $(CFLAGS) $(source) -o $@ $(LDFLAGS)

generate_dataset: src/generate_dataset.cpp | external/.stamp
	g++ -std=c++17 -O3 -DFMT_HEADER_ONLY -Iinclude -I$(CXXOPTS_INC) -I$(FMT_INC) -I$(ZIPF_INC) -I$(PROGRESS_INC) src/generate_dataset.cpp -o $@

DATASET = out_asc_uniform_1000_1000_100.bin \
          out_desc_uniform_1000_1000_100.bin

$(DATASET): generate_dataset
	./generate_dataset --iterations 100 --universe 1000 -k 1000

run: $(program) $(DATASET)
	$(LAUNCHER) ./$(program) --input out_asc_uniform_1000_1000_100.bin
	$(LAUNCHER) ./$(program) --input out_desc_uniform_1000_1000_100.bin

clean:
	rm -rf $(program) generate_dataset $(DATASET) external release
