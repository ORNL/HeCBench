# simplemoc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME simplemoc
    MODEL hip
    SOURCES init.cu io.cu main.cu
    CATEGORIES simulation
)
