# expdist-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME expdist
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
