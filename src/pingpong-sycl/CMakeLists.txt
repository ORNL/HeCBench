# pingpong-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pingpong
    MODEL sycl
    SOURCES main-mpi.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES pthread
)
