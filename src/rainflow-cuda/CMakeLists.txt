# rainflow-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rainflow
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
