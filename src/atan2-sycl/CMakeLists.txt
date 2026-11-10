# atan2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME atan2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
