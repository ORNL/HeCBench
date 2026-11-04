# keogh-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME keogh
    MODEL hip
    SOURCES main.cu
    CATEGORIES search
)
