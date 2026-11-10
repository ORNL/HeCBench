# fft-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fft
    MODEL cuda
    SOURCES main.cu
    CATEGORIES signal
)
