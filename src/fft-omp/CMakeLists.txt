# fft-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME fft
    MODEL omp
    SOURCES main.cpp
    CATEGORIES signal
)
