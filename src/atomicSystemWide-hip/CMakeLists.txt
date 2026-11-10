# atomicSystemWide-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicSystemWide
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
