# che-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME che
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
