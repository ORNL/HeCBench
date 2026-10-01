# bitcracker-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../bitcracker-cuda")

add_hecbench_benchmark(
    NAME bitcracker
    MODEL hip
    SOURCES attack.cu main.cu ${INC_DIR}/utils.cpp w_blocks.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES algorithms
)
