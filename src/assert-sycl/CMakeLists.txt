# assert-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME assert
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
