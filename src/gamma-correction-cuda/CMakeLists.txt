# gamma-correction-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gamma-correction
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
