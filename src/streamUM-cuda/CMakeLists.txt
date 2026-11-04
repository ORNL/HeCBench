# streamUM-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamUM
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
    COMPILE_OPTIONS -Xcompiler -fopenmp 
    LINK_LIBRARIES cublas -fopenmp
)
