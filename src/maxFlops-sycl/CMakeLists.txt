# maxFlops-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME maxFlops
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
