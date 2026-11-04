# mpc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mpc
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
