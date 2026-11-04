# langevin-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME langevin
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    COMPILE_OPTIONS -use_fast_math
)
