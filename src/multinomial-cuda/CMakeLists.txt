# multinomial-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME multinomial
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
