# colorwheel-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME colorwheel
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
