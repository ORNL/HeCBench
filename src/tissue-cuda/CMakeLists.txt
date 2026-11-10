# tissue-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tissue
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
