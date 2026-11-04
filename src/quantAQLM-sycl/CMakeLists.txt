# quantAQLM-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME quantAQLM
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
