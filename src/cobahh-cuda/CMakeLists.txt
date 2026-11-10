# cobahh-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME cobahh
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
