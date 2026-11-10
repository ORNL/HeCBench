# vote-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME vote
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
