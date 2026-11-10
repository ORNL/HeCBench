# michalewicz-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME michalewicz
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
