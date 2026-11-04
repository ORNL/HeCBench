# complex-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME complex
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
