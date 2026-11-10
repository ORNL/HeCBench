# lid-driven-cavity-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lid-driven-cavity
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
