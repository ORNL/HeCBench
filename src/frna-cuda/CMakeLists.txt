# frna-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME frna
    MODEL cuda
    SOURCES frna.cu main.c fbase.c fparam.c util.c
    CATEGORIES bioinformatics
    COMPILE_OPTIONS -DINT -use_fast_math
)
