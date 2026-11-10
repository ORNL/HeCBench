# ccsd-trpdrv-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../ccsd-trpdrv-cuda")

add_hecbench_benchmark(
    NAME ccsd-trpdrv
    MODEL hip
    SOURCES ccsd_tengy.cu ${INC_DIR}/ccsd_trpdrv.cu  ${INC_DIR}/main.cu
    CATEGORIES simulation
)
