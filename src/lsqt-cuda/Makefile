CC   = nvcc
ARCH = sm_60

# -DDEBUG sets a constant seed for random number generation
CFLAGS = -O3 -std=c++17 -arch=$(ARCH) -use_fast_math -DDEBUG


SOURCES = main.cu lsqt.cu sigma.cu vector.cu hamiltonian.cu model.cu \
	cu model_general.cu model_lattice.cu anderson.cu charge.cu
	
_OBJ = main.o lsqt.o sigma.o vector.o hamiltonian.o model.o model_general.o \
	model_lattice.o anderson.o charge.o

HEADERS = sigma.h vector.h lsqt.h hamiltonian.h model.h anderson.h charge.h common.h


ODIR = obj_gpu
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: builddir main

builddir:
	mkdir -p $(ODIR)

$(ODIR)/%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

main: $(OBJ)
	$(CC) $^ -o $@

clean:
	rm -f $(ODIR)/*.o main

run: main
	$(LAUNCHER) ./main examples/input.txt
