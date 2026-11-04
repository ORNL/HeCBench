# memcpy-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME memcpy
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
)
