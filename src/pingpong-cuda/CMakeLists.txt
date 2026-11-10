# pingpong-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pingpong
    MODEL cuda
    SOURCES main-mpi.cu
    CATEGORIES bandwidth
)
