# concurrentKernels-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME concurrentKernels
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
