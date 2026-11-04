# kernelLaunch-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME kernelLaunch
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
