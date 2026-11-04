# streamPriority-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamPriority
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
