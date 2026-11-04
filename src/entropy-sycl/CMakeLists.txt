# entropy-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME entropy
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES encoding
)
