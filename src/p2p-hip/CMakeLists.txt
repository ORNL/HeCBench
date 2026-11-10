# p2p-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME p2p
    MODEL hip
    SOURCES main.cu
    CATEGORIES bandwidth
)
