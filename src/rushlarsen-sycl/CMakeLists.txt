# rushlarsen-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rushlarsen
    MODEL sycl
    SOURCES main.cpp reference.cpp utils.cpp
    CATEGORIES simulation
)
