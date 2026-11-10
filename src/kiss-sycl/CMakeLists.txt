# kiss-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME kiss
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES encoding
)
