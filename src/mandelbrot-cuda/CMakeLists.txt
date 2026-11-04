# mandelbrot-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mandelbrot
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
