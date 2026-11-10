# bonds-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bonds
    MODEL cuda
    SOURCES bondsEngine.cu
    CATEGORIES finance
)
