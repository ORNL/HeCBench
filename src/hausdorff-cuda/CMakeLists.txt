# hausdorff-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hausdorff
    MODEL cuda
    SOURCES main.cu
    CATEGORIES geoscience
    COMPILE_OPTIONS --expt-relaxed-constexpr
)
