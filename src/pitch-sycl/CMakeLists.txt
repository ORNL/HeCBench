# pitch-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pitch
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
