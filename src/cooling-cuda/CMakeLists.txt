# cooling-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME cooling
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -use_fast_math
)
