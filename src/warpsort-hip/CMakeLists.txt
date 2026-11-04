# warpsort-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME warpsort
    MODEL hip
    SOURCES main.cpp warpsort.cu
    CATEGORIES sorting
    COMPILE_OPTIONS -DWARP_SIZE=32 -DHALF_WARP_SIZE=16
)
