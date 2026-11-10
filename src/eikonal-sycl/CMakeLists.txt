# eikonal-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../eikonal-cuda")

add_hecbench_benchmark(
    NAME eikonal
    MODEL sycl
    SOURCES StructuredEikonal.cpp fim.cpp kernel.cpp main.cpp ${INC_DIR}/timer.cpp
    CATEGORIES simulation
)
