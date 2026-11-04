# sosfil-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sosfil
    MODEL hip
    SOURCES main.cu
    CATEGORIES signal
)
