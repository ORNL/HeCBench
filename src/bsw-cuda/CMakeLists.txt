# bsw-cuda/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/test-data.tar.gz")

add_hecbench_benchmark(
    NAME bsw
    MODEL cuda
    SOURCES driver.cu main.cu utils.cu
    CATEGORIES bioinformatics
)
