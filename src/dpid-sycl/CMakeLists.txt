# dpid-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME dpid
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES imaging
)
