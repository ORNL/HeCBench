# megablocks-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME megablocks
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
