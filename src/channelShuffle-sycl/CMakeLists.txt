# channelShuffle-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME channelShuffle
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
