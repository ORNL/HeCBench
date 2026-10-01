#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp shrUtils.cpp cmd_arg_reader.cpp 

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -DUNIX -DBENCH_ITERATIONS=100 -std=c++17 -Wall

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o : main.cpp \
         tridiagonal.h\
         shrUtils.h\
         file_read_write.h\
         test_gen_result_check.h\
         cpu_solvers.h\
         pcr_small_systems.h\
         cyclic_small_systems.h\
         sweep_small_systems.h\
         pcr_kernels.cpp\
         cyclic_kernels.cpp\
         sweep_kernels.cpp
	$(CC) $(CFLAGS) -c $< -o $@

shrUtils.o: shrUtils.cpp shrUtils.h cmd_arg_reader.h exception.h
	$(CC) $(CFLAGS) -c $< -o $@

cmd_arg_reader.o: cmd_arg_reader.cpp cmd_arg_reader.h exception.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.dat *.timing oclTridiagonal.txt SdkConsoleLog.txt *.csv

run: $(program)
	$(LAUNCHER) ./$(program) -num_systems=524288
	$(LAUNCHER) ./$(program) --lmem -num_systems=524288
	$(LAUNCHER) ./$(program) --vec4 -num_systems=524288 
