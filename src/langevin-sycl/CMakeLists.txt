# langevin-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME langevin
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    COMPILE_OPTIONS -ffast-math
)
