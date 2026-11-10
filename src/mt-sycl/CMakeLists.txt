# mt-sycl/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/data.tar.gz")

add_hecbench_benchmark(
    NAME mt
    MODEL sycl
    SOURCES MT.cpp MT_gold.cpp genmtrand.cpp
    CATEGORIES rng
)
