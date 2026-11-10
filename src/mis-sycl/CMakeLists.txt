# mis-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mis
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES graph
)
