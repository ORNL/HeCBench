# feynman-kac-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME feynman-kac
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
