# mandelbrot-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mandelbrot
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
