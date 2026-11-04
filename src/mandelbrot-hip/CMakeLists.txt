# mandelbrot-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mandelbrot
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
