# qrg-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME qrg
    MODEL hip
    SOURCES main.cu reference.cu
    CATEGORIES rng
)
