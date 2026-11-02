# vdi-sycl/CMakeLists.txt

set(VDN_DELTA_SYCL_LIBS pthread)
set(VDN_DELTA_SYCL_OPTIONS)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    list(APPEND VDN_DELTA_SYCL_OPTIONS -DWDN_ONEMKL -qmkl=parallel)
    list(APPEND VDN_DELTA_SYCL_LIBS -qmkl=parallel)
endif()

add_hecbench_benchmark(
    NAME vdi
    MODEL sycl
    SOURCES main.cpp fused.cpp libraries.cpp
    CATEGORIES math
    COMPILE_OPTIONS ${VDN_DELTA_SYCL_OPTIONS}
    LINK_LIBRARIES ${VDN_DELTA_SYCL_LIBS}
)
