# mdh-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mdh
    MODEL sycl
    SOURCES WKFUtils.cpp main.cpp
    CATEGORIES simulation
)
