# rsc-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../rsc-cuda")

add_hecbench_benchmark(
    NAME rsc
    MODEL sycl
    SOURCES main.cpp model_eval.cpp ${INC_DIR}/model_fitting.cpp
    CATEGORIES math
    LINK_LIBRARIES pthread
)
