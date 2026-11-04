# su3-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME su3
    MODEL cuda
    SOURCES su3_nn_bench.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -DMILC_COMPLEX
)
