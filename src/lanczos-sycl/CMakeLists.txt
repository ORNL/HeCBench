# lanczos-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lanczos
    MODEL sycl
    SOURCES lanczos.cpp main.cpp
    CATEGORIES math
)
