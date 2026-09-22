# pingpong-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME pingpong
    MODEL hip
    SOURCES main-mpi.cu
    CATEGORIES algorithms
    LINK_LIBRARIES pthread
)
