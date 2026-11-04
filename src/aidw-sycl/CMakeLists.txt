# aidw-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME aidw
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES geoscience
    COMPILE_OPTIONS -ffast-math
)
