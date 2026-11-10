# reaction-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME reaction
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
