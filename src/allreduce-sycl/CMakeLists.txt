# allreduce-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME allreduce
    MODEL sycl
    SOURCES collectives.cpp main.cpp timer.cpp
    CATEGORIES bandwidth
)
