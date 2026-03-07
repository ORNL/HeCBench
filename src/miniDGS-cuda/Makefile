N       = 6
# must be greater than 1 (README)
NP      = 2  

# define variables
VPATH   = ./
HDRDIR  = ./include

# adjust this for your system
CUDADIR=/auto/software/nvidia/x86_64/cuda-11.0
OPENMPI_BINDIR=/usr/lib64/openmpi3/bin

# set options for this machine
# specify which compilers to use for c, fortran and linking
CC	= $(OPENMPI_BINDIR)/mpicc
LD	= $(OPENMPI_BINDIR)/mpicc
NVCC	= $(CUDADIR)/bin/nvcc
#DEBUG   = -g -G -DDEBUG
FAST    = -O3 --use_fast_math 
LAUNCHER=

# compiler flags to be used (set to compile with debugging on)
CFLAGS = -Dp_N=$(N) -DNDG3d -DCUDA -I$(HDRDIR) -O3 -I$(CUDADIR)/include

# link flags to be used 
LDFLAGS	= -I$(HDRDIR) -O3 

# libraries to be linked in
LIBS	= -lm -L./3rdParty/ParMetis-3.1/ -lparmetis -lmetis -L$(CUDADIR)/lib64 -lcuda -lcudart

# types of files we are going to construct rules for
.SUFFIXES: .c .f .cu

# rule for .c files
.c.o:
	$(CC) $(CFLAGS) -o $*.o -c $*.c
.cu.o: 
	$(NVCC) -Xcompiler -Wall $(CFLAGS) $(DEBUG) $(FAST) -o $*.o -c $*.cu  

# list of objects to be compiled
OBJS    = \
	src/Mesh3d.o\
	src/Utils.o\
	src/LoadBalance3d.o\
	src/FacePair3d.o\
	src/ParallelPairs.o\
	src/BuildMaps3d.o\
	src/StartUp3d.o\
	src/MaxwellsRun3d.o\
	src/MaxwellsMPI3d.o\
	src/MaxwellsKernel3d.o\
	src/MaxwellsDriver3d.o

main:$(OBJS)
	$(LD) $(LDFLAGS) -o main $(OBJS) $(LIBS)

# what to do if user types "make clean"
clean :
	rm -rf $(OBJS) main
	
veryclean :
	make clean
	cd 3rdParty/ParMetis-3.1 && make clean

run :
	mpiexec -n $(NP) ./main  Meshes/FS_K01022.neu
