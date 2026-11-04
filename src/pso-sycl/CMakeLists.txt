# pso-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../pso-cuda")

add_hecbench_benchmark(
    NAME pso
    MODEL sycl
    SOURCES kernel_gpu.cpp ${INC_DIR}/kernel_cpu.cpp ${INC_DIR}/main.cpp
    CATEGORIES simulation
)
