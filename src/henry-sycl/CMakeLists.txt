# henry-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME henry
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -ffast-math
)
