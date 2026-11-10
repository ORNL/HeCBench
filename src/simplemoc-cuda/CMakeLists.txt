# simplemoc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME simplemoc
    MODEL cuda
    SOURCES init.cu io.cu main.cu
    CATEGORIES simulation
)
