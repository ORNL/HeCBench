# testSNAP-omp/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/refdata.tar.gz")

add_hecbench_benchmark(
    NAME testSNAP
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -DREFDATA_TWOJ=14
)
