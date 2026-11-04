# mmcsf-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mmcsf-cuda")

add_hecbench_benchmark(
    NAME mmcsf
    MODEL hip
    SOURCES kernels.cu ${INC_DIR}/main.cu ${INC_DIR}/mttkrp_cpu.cu
    CATEGORIES ml
    INCLUDE_DIRS ${Boost_INCLUDE_DIRS}
    LINK_LIBRARIES pthread
)
