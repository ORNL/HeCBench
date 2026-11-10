# morphology-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME morphology
    MODEL hip
    SOURCES main.cu morphology.cu
    CATEGORIES imaging
)
