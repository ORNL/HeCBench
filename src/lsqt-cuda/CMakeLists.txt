# lsqt-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/examples.tar.gz")

add_hecbench_benchmark(
    NAME lsqt
    MODEL cuda
    SOURCES main.cu lsqt.cu sigma.cu vector.cu hamiltonian.cu model.cu model_general.cu model_lattice.cu anderson.cu charge.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -use_fast_math -DDEBUG
)
