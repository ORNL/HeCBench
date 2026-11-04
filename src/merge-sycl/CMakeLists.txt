# merge-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME merge
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES sorting
)
