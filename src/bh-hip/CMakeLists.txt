# bh-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bh
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
