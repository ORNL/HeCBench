# hotspot3D-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/../data/hotspot3D/hotspot3D.tar.bz")

add_hecbench_benchmark(
    NAME hotspot3D
    MODEL cuda
    SOURCES 3D.cu 3D_helper.cu
    CATEGORIES simulation
)
