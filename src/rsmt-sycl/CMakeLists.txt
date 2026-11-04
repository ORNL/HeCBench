# rsmt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsmt
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES graph
)
