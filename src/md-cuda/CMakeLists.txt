# md-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME md
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
