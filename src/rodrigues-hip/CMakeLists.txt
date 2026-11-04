# rodrigues-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rodrigues
    MODEL hip
    SOURCES main.cu
    CATEGORIES robotics
)
