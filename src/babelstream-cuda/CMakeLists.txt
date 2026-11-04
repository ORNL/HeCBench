# babelstream-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME babelstream
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
)
