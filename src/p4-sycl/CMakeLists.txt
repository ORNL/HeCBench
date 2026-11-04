# p4-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME p4
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
    COMPILE_OPTIONS -ffast-math
)
