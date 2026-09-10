# swiglu-oai-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME swiglu-oai
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
