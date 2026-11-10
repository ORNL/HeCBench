# channelShuffle-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME channelShuffle
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
