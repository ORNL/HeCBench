# fluidSim-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../fluidSim-cuda")

add_hecbench_benchmark(
    NAME fluidSim
    MODEL hip
    SOURCES kernels.cu ${INC_DIR}/main.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -DVERIFY
)
