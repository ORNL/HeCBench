# saxpy-ompt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME saxpy-ompt
    MODEL sycl
    SOURCES saxpy.c hsaxpy.c asaxpy.c check1ns.c wtcalc.c
    CATEGORIES algorithms
    COMPILE_OPTIONS -fopenmp -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
    LINK_LIBRARIES -fiopenmp -fopenmp-targets=spir64 -qmkl=parallel 
    C_AS_CXX
)
