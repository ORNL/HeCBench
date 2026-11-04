# gpp-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gpp
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
