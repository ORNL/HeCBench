# kda-sycl/CMakeLists.txt

# Host OpenMP for the CPU reference. DPC++ needs -Xarch_host so the flag is
# not forwarded to the device compiler; AdaptiveCpp/hipSYCL reject that flag.
if(CMAKE_CXX_COMPILER_ID STREQUAL "AdaptiveCpp" OR
   CMAKE_CXX_COMPILER MATCHES "acpp|syclcc" OR
   (DEFINED SYCL_COMPILER_TYPE AND SYCL_COMPILER_TYPE MATCHES "hipSYCL|AdaptiveCpp"))
  set(KDA_HOST_OPENMP -fopenmp)
else()
  set(KDA_HOST_OPENMP -Xarch_host -fopenmp)
endif()

add_hecbench_benchmark(
    NAME kda
    MODEL sycl
    SOURCES main.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../kda-cuda
    CATEGORIES ml
    COMPILE_OPTIONS ${KDA_HOST_OPENMP}
    LINK_LIBRARIES ${KDA_HOST_OPENMP}
    TEST_REGEX "PASS"
    TEST_ARGS 1 16 2
)
