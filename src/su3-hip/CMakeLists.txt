# su3-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME su3
    MODEL hip
    SOURCES su3_nn_bench.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -DMILC_COMPLEX
)
