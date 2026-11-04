# lr-sycl/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/assets.tar.gz")

add_hecbench_benchmark(
    NAME lr
    MODEL sycl
    SOURCES linear_iter.cpp linear_par.cpp main.cpp
    CATEGORIES ml
)
