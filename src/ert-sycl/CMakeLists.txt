# ert-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ert
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
