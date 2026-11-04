# grep-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME grep
    MODEL hip
    SOURCES main.cu nfautil.cu regex.cu
    CATEGORIES search
)
