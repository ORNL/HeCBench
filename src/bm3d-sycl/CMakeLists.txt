# bm3d-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bm3d
    MODEL sycl
    SOURCES blockmatching.cpp dct8x8.cpp filtering.cpp main.cpp
    CATEGORIES imaging
    LINK_LIBRARIES pthread
)
