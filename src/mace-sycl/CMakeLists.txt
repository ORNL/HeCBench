# mace-sycl/CMakeLists.txt

if(HECBENCH_ENABLE_SYCL)
    find_package(MKL CONFIG QUIET
        PATHS /opt/intel/oneapi/mkl/latest/lib/cmake/mkl)
    if(NOT TARGET MKL::MKL_SYCL)
        message(WARNING "Skipping mace-sycl: oneMKL SYCL (MKL::MKL_SYCL) not found")
        return()
    endif()
endif()

add_hecbench_benchmark(
    NAME mace
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
    COMPILE_OPTIONS -DUSE_ONEMKL
    LINK_LIBRARIES MKL::MKL_SYCL
)
