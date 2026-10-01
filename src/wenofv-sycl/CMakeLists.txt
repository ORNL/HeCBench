# wenofv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME wenofv
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
