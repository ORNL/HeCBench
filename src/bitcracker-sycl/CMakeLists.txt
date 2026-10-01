# bitcracker-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../bitcracker-cuda")

add_hecbench_benchmark(
    NAME bitcracker
    MODEL sycl
    SOURCES attack.cpp main.cpp ${INC_DIR}/utils.cpp w_blocks.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffast-math
)
