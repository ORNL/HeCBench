# ccsd-trpdrv-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME ccsd-trpdrv
    MODEL omp
    SOURCES ccsd_tengy.cpp ccsd_trpdrv.cpp main.cpp
    CATEGORIES simulation
)
