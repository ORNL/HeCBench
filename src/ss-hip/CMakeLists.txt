# ss-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ss
    MODEL hip
    SOURCES main.cu
    CATEGORIES search
)
