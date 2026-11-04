# pns-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pns
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
