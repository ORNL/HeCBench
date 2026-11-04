# frna-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME frna
    MODEL sycl
    SOURCES frna.cpp main.c fbase.c fparam.c util.c
    CATEGORIES bioinformatics
    COMPILE_OPTIONS -DINT -ffast-math
)
