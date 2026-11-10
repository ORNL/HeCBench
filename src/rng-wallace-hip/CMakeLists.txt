# rng-wallace-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rng-wallace
    MODEL hip
    SOURCES main.cu
    CATEGORIES rng
)
