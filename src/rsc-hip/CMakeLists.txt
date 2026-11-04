# rsc-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../rsc-cuda")

add_hecbench_benchmark(
    NAME rsc
    MODEL hip
    SOURCES main.cu model_eval.cu ${INC_DIR}/model_fitting.cpp
    CATEGORIES math
    LINK_LIBRARIES pthread
)
