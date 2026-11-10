# cfd-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/../data/cfd/cfd.tar.bz")

add_hecbench_benchmark(
    NAME cfd
    MODEL cuda
    SOURCES euler3d.cu
    CATEGORIES simulation
)
