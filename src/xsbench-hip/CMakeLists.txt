# xsbench-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../xsbench-cuda")

add_hecbench_benchmark(
    NAME xsbench
    MODEL hip
    SOURCES Simulation.cu
            ${INC_DIR}/Main.cu
            ${INC_DIR}/io.cu
            ${INC_DIR}/GridInit.cu
            ${INC_DIR}/Materials.cu
            ${INC_DIR}/XSutils.cu
    CATEGORIES simulation
)
