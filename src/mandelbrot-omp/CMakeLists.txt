# mandelbrot-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mandelbrot
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
