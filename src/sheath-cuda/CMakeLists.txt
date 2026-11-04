# sheath-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sheath
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
