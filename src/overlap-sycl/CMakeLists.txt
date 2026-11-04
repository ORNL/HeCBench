# overlap-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME overlap
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
