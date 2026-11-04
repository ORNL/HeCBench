# axpby-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME axpby
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
