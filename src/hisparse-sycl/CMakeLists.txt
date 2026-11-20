# hisparse-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hisparse
    MODEL sycl
    SOURCES main.cpp hisparse_kernels.cpp
    CATEGORIES ml
    TEST_ARGS 8 256 1
    TEST_REGEX "PASS"
)

# One translation unit: with separately compiled objects, DPC++'s amdgcn
# device link drops the kernel info and every launch asserts.
if(TARGET hisparse-sycl)
    set_target_properties(hisparse-sycl PROPERTIES UNITY_BUILD ON)
endif()
