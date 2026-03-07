CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu

CFLAGS +=-O3 -DDEBUG -Wall -Minfo -mp=gpu -gpu=$(SM),fastmath

ifeq ($(DEVICE),gpu)
  CFLAGS +=-O3 -DDEBUG -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-O3 -DDEBUG -qopenmp
endif


SOURCES = main.cpp lsqt.cpp sigma.cpp vector.cpp hamiltonian.cpp model.\
	cpp model_general.cpp model_lattice.cpp anderson.cpp charge.cpp
	
ifdef OS # for Windows
_OBJ = main.obj lsqt.obj sigma.obj vector.obj hamiltonian.obj model.obj model_general.obj \
	model_lattice.obj anderson.obj charge.obj
else # for Linux
_OBJ = main.o lsqt.o sigma.o vector.o hamiltonian.o model.o model_general.o \
	model_lattice.o anderson.o charge.o
endif

HEADERS = sigma.h vector.h lsqt.h hamiltonian.h model.h anderson.h charge.h common.h


ODIR = obj_gpu
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: main

$(ODIR)/%.o: %.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

main: $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f $(ODIR)/*.o main *.out

run: main
	$(LAUNCHER) ./main ../lsqt-cuda/examples/input.txt
