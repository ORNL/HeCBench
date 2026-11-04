# bsw-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bsw
    MODEL hip
    SOURCES driver.cu main.cu utils.cu
    CATEGORIES bioinformatics
)
