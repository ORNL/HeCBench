# gsplat4d-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../gsplat4d-cuda")

add_hecbench_benchmark(
    NAME gsplat4d
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    INCLUDE_DIRS ${INC_DIR}
)
