# warpsort-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME warpsort
    MODEL sycl
    SOURCES main.cpp warpsort.cpp
    CATEGORIES sorting
    COMPILE_OPTIONS -DWARP_SIZE=32 -DHALF_WARP_SIZE=16
)
