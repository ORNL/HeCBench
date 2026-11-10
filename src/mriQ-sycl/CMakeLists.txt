# mriQ-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mriQ-cuda")

add_hecbench_benchmark(
    NAME mriQ
    MODEL sycl
    SOURCES main.cpp ${INC_DIR}/file.cc
    CATEGORIES imaging
    INCLUDE_DIRS ${INC_DIR}
    COMPILE_OPTIONS -ffast-math
)
