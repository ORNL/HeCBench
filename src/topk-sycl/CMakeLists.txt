# topk-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../blockScan-sycl/utils")

add_hecbench_benchmark(
    NAME topk
    MODEL sycl
    SOURCES main.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES algorithms
)

# ptxas runs at link. Cap registers so the 1024-thread kernel can launch.
if(TARGET topk-sycl AND HECBENCH_SYCL_TARGET MATCHES "nvptx")
    target_link_options(topk-sycl PRIVATE
        "SHELL:-Xcuda-ptxas --maxrregcount=64")
endif()
