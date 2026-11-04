# qrg-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME qrg
    MODEL cuda
    SOURCES main.cu reference.cu
    CATEGORIES rng
)
