# tpacf-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../tpacf-cuda")

add_hecbench_benchmark(
    NAME tpacf
    MODEL hip
    SOURCES ${INC_DIR}/main.cu ${INC_DIR}/args.cu compute.cu
    CATEGORIES simulation
)
