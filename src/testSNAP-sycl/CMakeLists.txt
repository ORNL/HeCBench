# testSNAP-sycl/CMakeLists.txt

set(DATA_DIR "${CMAKE_CURRENT_LIST_DIR}/../testSNAP-omp")
UNZIPFILE("${DATA_DIR}/refdata.tar.gz")

add_hecbench_benchmark(
    NAME testSNAP
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    INCLUDE_DIRS ${DATA_DIR}
    COMPILE_OPTIONS -DREFDATA_TWOJ=14
)
