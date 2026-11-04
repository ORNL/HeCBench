# langford-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME langford
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
