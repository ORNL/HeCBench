# gamma-correction-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gamma-correction
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
