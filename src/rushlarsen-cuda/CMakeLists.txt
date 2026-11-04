# rushlarsen-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rushlarsen
    MODEL cuda
    SOURCES main.cu reference.cu utils.cu
    CATEGORIES simulation
)
