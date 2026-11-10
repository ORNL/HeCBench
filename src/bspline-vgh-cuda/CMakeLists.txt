# bspline-vgh-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bspline-vgh
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
