# car-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME car
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
