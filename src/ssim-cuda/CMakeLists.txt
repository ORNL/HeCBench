# ssim-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ssim
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
