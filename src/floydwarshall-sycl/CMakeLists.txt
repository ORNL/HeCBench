# floydwarshall-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME floydwarshall
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES graph
)
