# lulesh-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

add_hecbench_benchmark(
    NAME lulesh
    MODEL sycl
    SOURCES lulesh-comm.cc lulesh-init.cc lulesh-util.cc lulesh-viz.cc lulesh.cc
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES simulation
)
