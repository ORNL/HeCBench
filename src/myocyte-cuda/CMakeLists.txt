# myocyte-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME myocyte
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
