# heat-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME heat
    MODEL cuda
    SOURCES heat.cu
    CATEGORIES simulation
)
