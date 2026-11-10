# distort-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME distort
    MODEL cuda
    SOURCES distort.cu main.cu
    CATEGORIES imaging
)
