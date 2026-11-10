# bmf-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bmf
    MODEL hip
    SOURCES src/main.cu
    COMPILE_OPTIONS --default-stream=per-thread
    LINK_LIBRARIES -lgomp
    CATEGORIES ml
)
