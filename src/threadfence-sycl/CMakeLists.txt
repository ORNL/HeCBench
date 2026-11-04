# threadfence-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME threadfence
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
