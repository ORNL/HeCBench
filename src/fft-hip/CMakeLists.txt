# fft-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME fft
    MODEL hip
    SOURCES main.cu
    CATEGORIES signal
)
