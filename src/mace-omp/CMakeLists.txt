# mace-omp/CMakeLists.txt
# Vendor BLAS is selected from the OpenMP offload compiler, matching the
# Makefiles: icpx → oneMKL, nvc++ → cuBLAS, amdclang++ → rocBLAS.

set(_mace_omp_compile_options)
set(_mace_omp_link_libraries)
set(_mace_omp_include_dirs)

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  list(APPEND _mace_omp_compile_options -DMACE_OMP_ONEMKL)
  list(APPEND _mace_omp_link_libraries -qmkl)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" OR CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
  list(APPEND _mace_omp_compile_options -DMACE_OMP_CUBLAS)
  list(APPEND _mace_omp_link_libraries cublas)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND _mace_omp_compile_options -DMACE_OMP_ROCBLAS -D__HIP_PLATFORM_AMD__)
  if(DEFINED ENV{ROCM_PATH})
    set(_mace_rocm "$ENV{ROCM_PATH}")
  else()
    set(_mace_rocm /opt/rocm)
  endif()
  list(APPEND _mace_omp_include_dirs "${_mace_rocm}/include")
  list(APPEND _mace_omp_link_libraries
       "-L${_mace_rocm}/lib" "-Wl,-rpath,${_mace_rocm}/lib" rocblas amdhip64)
endif()

add_hecbench_benchmark(
    NAME mace
    MODEL omp
    SOURCES main.cpp
    CATEGORIES machine-learning
    COMPILE_OPTIONS ${_mace_omp_compile_options}
    LINK_LIBRARIES ${_mace_omp_link_libraries}
    INCLUDE_DIRS ${_mace_omp_include_dirs}
)
