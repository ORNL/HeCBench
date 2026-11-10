# nosync-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME nosync
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
    LINK_LIBRARIES pthread
)
