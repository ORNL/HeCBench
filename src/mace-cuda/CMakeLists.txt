# mace-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mace
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    COMPILE_OPTIONS -Xcompiler -fopenmp
    LINK_LIBRARIES cublas -fopenmp
)
