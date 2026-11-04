# lulesh-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lulesh
    MODEL cuda
    SOURCES lulesh-init.cu lulesh-util.cu lulesh-viz.cu lulesh.cu
    CATEGORIES simulation
)
