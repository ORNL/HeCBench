# particlefilter-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME particlefilter
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
