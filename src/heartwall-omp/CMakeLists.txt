# heartwall-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME heartwall
    MODEL omp
    SOURCES main.cpp kernel/kernel.cpp util/avi/avilib.c util/avi/avimod.c util/file/file.c
    INCLUDE_DIRS util/avi util/file
    CATEGORIES simulation
)
