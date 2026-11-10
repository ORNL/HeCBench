# rushlarsen-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rushlarsen
    MODEL hip
    SOURCES main.cu reference.cu utils.cu
    CATEGORIES simulation
)
