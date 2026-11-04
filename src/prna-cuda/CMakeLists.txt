# prna-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME prna
    MODEL cuda
    SOURCES prna.cu main.c base.c param.c util.c
    CATEGORIES bioinformatics
    COMPILE_OPTIONS -DFLOAT -use_fast_math
)
