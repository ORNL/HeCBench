# goulash-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME goulash
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
