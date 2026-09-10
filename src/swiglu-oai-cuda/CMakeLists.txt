# swiglu-oai-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME swiglu-oai
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
