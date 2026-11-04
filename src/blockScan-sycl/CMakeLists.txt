# blockScan-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME blockScan
    MODEL sycl
    SOURCES main.cpp
    INCLUDE_DIRS utils
    COMPILE_OPTIONS -fno-sycl-id-queries-fit-in-int
    CATEGORIES language
)
