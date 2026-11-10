# ne-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ne
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
