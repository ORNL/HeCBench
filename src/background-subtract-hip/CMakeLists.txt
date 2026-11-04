# background-subtract-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME background-subtract
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
