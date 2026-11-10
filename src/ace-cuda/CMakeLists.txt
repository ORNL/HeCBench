# ace-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ace
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
