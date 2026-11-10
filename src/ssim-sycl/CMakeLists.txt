# ssim-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ssim
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
