# channelSum-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME channelSum
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
