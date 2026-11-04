# bgmv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bgmv
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
