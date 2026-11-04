# mriQ-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mriQ-cuda")

add_hecbench_benchmark(
    NAME mriQ
    MODEL omp
    SOURCES main.cpp ${INC_DIR}/file.cc
    ENABLE_FASTMATH true
    CATEGORIES imaging
)
