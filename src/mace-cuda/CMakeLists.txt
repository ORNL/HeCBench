# mace-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mace
    MODEL cuda
    SOURCES main.cu
    CATEGORIES machine-learning
    LINK_LIBRARIES cublas
)
