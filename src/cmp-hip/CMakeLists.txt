# cmp-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../cmp-cuda")

add_hecbench_benchmark(
    NAME cmp
    MODEL hip
    SOURCES main.cu
            ${INC_DIR}/log.cpp
            ${INC_DIR}/parser.cpp
            ${INC_DIR}/su_cdp.cpp
            ${INC_DIR}/su_gather.cpp
            ${INC_DIR}/su_trace.cpp 
            ${INC_DIR}/reference.cpp
    CATEGORIES simulation
)
