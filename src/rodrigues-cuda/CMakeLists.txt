# rodrigues-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rodrigues
    MODEL cuda
    SOURCES main.cu
    CATEGORIES robotics
)
