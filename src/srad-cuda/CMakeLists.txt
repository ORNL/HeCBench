# srad-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/../data/srad/srad.tar.bz")

add_hecbench_benchmark(
    NAME srad
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
