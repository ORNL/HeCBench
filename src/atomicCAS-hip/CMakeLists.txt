# atomicCAS-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicCAS
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
