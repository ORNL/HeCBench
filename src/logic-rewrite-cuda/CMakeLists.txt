# logic-rewrite-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME logic-rewrite
    MODEL cuda
    SOURCES aig_manager.cu balance.cu command_manager.cu main.cu print.cu refactor.cu refactor_core.cu refactor_mffc.cu rewrite.cu strash.cu truth.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -use_fast_math
    LINK_LIBRARIES pthread
)
