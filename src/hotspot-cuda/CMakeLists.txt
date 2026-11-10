# hotspot-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/../data/hotspot/hotspot.tar.bz")

add_hecbench_benchmark(
    NAME hotspot
    MODEL cuda
    SOURCES hotspot.cu
    CATEGORIES simulation
)
