# feynman-kac-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME feynman-kac
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
