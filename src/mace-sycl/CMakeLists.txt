# mace-sycl/CMakeLists.txt

if(HECBENCH_ENABLE_SYCL)
    find_package(MKL CONFIG REQUIRED
        PATHS /opt/intel/oneapi/mkl/latest/lib/cmake/mkl)
endif()

add_hecbench_benchmark(
    NAME mace
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES machine-learning
    COMPILE_OPTIONS -DUSE_ONEMKL
    LINK_LIBRARIES MKL::MKL_SYCL
)
