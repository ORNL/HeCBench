# frna-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME frna
    MODEL omp
    SOURCES frna.cpp main.c fbase.c fparam.c util.c
    COMPILE_OPTIONS -DINT
    ENABLE_FASTMATH true
    CATEGORIES bioinformatics
)
