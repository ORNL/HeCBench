# nlll-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME nlll
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
