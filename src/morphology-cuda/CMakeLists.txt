# morphology-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME morphology
    MODEL cuda
    SOURCES main.cu morphology.cu
    CATEGORIES imaging
)
