# ring-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ring
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
