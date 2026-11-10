# sph-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sph
    MODEL cuda
    SOURCES fileio.cu fluid.cu geometry.cu
    CATEGORIES simulation
)
