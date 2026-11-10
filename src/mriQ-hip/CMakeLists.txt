# mriQ-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mriQ-cuda")

add_hecbench_benchmark(
    NAME mriQ
    MODEL hip
    SOURCES main.cu ${INC_DIR}/file.cc
    CATEGORIES imaging
    COMPILE_OPTIONS -ffast-math
)
