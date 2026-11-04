# p2p-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME p2p
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
)
