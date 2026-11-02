# prna-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME prna
    MODEL sycl
    SOURCES prna.cpp main.c base.c param.c util.c
    CATEGORIES algorithms
    COMPILE_OPTIONS -D__CUDACC__ -DFLOAT -ffast-math
    C_AS_CXX
)
