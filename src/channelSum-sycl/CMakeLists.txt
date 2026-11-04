# channelSum-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME channelSum
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
