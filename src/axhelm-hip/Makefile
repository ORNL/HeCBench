#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = main.o meshBasis.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -Ddfloat=float -Ddlong=int \
          -I../axhelm-cuda

# Linker Flags
FC = gfortran

# Resolved on demand, so goals that do not link need no Fortran compiler.
# $(FC) echoes the bare name when it cannot resolve the library, hence /%.
LIBGFORTRAN_PATH = $(filter /%,$(shell $(FC) -print-file-name=libgfortran.so))
LIBGFORTRAN = $(or $(wildcard $(LIBGFORTRAN_PATH)),\
              $(error Could not find libgfortran using $(FC)))

LDLIBS = BlasLapack/libBlasLapack.a $(LIBGFORTRAN)

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# Shared DVC archive lives in axhelm-cuda
BLAS_TGZ := ../axhelm-cuda/BlasLapack.tar.gz
BLAS_MK  := BlasLapack/Makefile

#===============================================================================
# Targets to Build
#===============================================================================
$(program): $(obj) BlasLapack/libBlasLapack.a 
	$(CC) $(CFLAGS) $+ -o $@ $(LDFLAGS) $(LDLIBS)

main.o : main.cu axhelmKernel.cpp ../axhelm-cuda/axhelmReference.cpp
	$(CC) $(CFLAGS) -c $< -o $@
	
meshBasis.o : meshBasis.cpp meshBasis.hpp
	$(CC) $(CFLAGS) -c $< -o $@

BlasLapack/libBlasLapack.a: $(BLAS_MK)
	$(MAKE) -C BlasLapack -j8

$(BLAS_MK):
	@test -f $(BLAS_TGZ) || { \
	  echo "Missing $(BLAS_TGZ). From the repository root run: dvc pull src/axhelm-cuda/BlasLapack.tar.gz" >&2; \
	  exit 1; }
	tar -xzf $(BLAS_TGZ)

clean:
	if [ -d BlasLapack ]; then cd BlasLapack && make clean; fi
	rm -rf $(program) $(obj)

# run one- and three-dimensional kernels
run: $(program)
	$(LAUNCHER) ./$(program) 1 8000 100
	$(LAUNCHER) ./$(program) 3 8000 100
