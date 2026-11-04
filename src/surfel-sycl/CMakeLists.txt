# surfel-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME surfel
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
