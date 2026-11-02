# wenofv-omp/CMakeLists.txt
#
# Disable FP contraction so the CPU reference and the offloaded kernels
# (both from kernels.h) agree. nvc++ uses -Mnofma (Makefile.nvc); clang,
# gcc, and icpx use -ffp-contract=off.

if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
    set(WENOFV_FP_CONTRACT -Mnofma)
else()
    set(WENOFV_FP_CONTRACT -ffp-contract=off)
endif()

add_hecbench_benchmark(
    NAME wenofv
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS ${WENOFV_FP_CONTRACT}
)
