# babelstream-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME babelstream
    MODEL hip
    SOURCES main.cu
    CATEGORIES bandwidth
)
