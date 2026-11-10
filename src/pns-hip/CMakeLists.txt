# pns-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME pns
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
