# trimul-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME trimul
    MODEL sycl
    SOURCES main.cpp
    COMPILE_OPTIONS -DUSE_GPU -Xarch_host -fopenmp
    LINK_LIBRARIES -Xarch_host -fopenmp
    CATEGORIES ml
)
