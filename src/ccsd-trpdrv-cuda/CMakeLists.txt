# ccsd-trpdrv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ccsd-trpdrv
    MODEL cuda
    SOURCES ccsd_tengy.cu ccsd_trpdrv.cu main.cu
    CATEGORIES simulation
)
