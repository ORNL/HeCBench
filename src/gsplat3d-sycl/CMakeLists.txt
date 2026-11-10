# gsplat3d-sycl/CMakeLists.txt

set(CPU_REFERENCE "${CMAKE_CURRENT_LIST_DIR}/../gsplat3d-cuda/reference.cpp")

add_hecbench_benchmark(
    NAME gsplat3d
    MODEL sycl
    SOURCES ${CPU_REFERENCE} main.cpp forward.cpp baseline.cpp optimized.cpp
    CATEGORIES imaging
    COMPILE_OPTIONS -Wno-incorrect-sub-group-size
    TEST_ARGS 4096 128 128 10
    TEST_REGEX "Rasterize backward: PASS"
    TEST_TIMEOUT 300
)

# The sources are compiled separately and linked. On AMD the old offload
# driver links a binary whose kernels the runtime cannot find.
if(TARGET gsplat3d-sycl AND HECBENCH_SYCL_TARGET MATCHES "nvptx|amdgcn")
    target_compile_options(gsplat3d-sycl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:--offload-new-driver>)
    target_link_options(gsplat3d-sycl PRIVATE --offload-new-driver)
endif()
