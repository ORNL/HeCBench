# sortKV-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sortKV
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES sorting
)
