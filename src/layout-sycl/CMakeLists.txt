# layout-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME layout
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
