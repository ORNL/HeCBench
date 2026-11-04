# sparkler-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sparkler
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
    LINK_LIBRARIES cublas
)
