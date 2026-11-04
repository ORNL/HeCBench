# sptrsv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sptrsv
    MODEL sycl
    SOURCES main.cpp sptrsv_syncfree.cpp
    CATEGORIES math
    COMPILE_OPTIONS -DWARP_PER_BLOCK=8 -DVALUE_TYPE=double -DBENCH_REPEAT=100 
)
