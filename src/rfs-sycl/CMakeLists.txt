# rfs-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rfs
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
