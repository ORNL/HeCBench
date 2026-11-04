# asta-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME asta
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
