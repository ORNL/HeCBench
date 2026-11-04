# atomicIntrinsics-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicIntrinsics
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
