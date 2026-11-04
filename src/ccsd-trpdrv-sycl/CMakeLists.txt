# ccsd-trpdrv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ccsd-trpdrv
    MODEL sycl
    SOURCES ccsd_tengy.cpp ccsd_trpdrv.cpp main.cpp
    CATEGORIES simulation
)
