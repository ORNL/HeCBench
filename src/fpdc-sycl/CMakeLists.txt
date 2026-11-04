# fpdc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME fpdc
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
