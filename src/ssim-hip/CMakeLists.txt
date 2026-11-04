# ssim-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ssim
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
