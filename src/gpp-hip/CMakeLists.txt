# gpp-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gpp
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
