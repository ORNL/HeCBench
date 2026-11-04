# bonds-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bonds
    MODEL hip
    SOURCES bondsEngine.cu
    CATEGORIES finance
)
