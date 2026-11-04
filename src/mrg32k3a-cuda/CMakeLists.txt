# mrg32k3a-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mrg32k3a
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES curand
)
