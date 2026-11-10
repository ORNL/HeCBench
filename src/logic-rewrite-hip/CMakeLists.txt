# logic-rewrite-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME logic-rewrite
    MODEL hip
    SOURCES aig_manager.cu balance.cu command_manager.cu main.cu print.cu refactor.cu refactor_core.cu refactor_mffc.cu rewrite.cu strash.cu truth.cu
    COMPILE_OPTIONS -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -ffast-math
    CATEGORIES simulation
    LINK_LIBRARIES pthread
)
