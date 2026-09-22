# leukocyte-omp/CMakeLists.txt

SET(MESCHACH_TARGETS copy.c err.c matrixio.c memory.c vecop.c 
                     matop.c pxop.c submat.c init.c otherio.c 
                     machine.c matlab.c ivecop.c version.c 
                     meminfo.c memstat.c lufactor.c bkpfacto.c 
                     chfactor.c qrfactor.c solve.c hsehldr.c 
                     givens.c update.c norm.c hessen.c symmeig.c 
                     schur.c svd.c fft.c mfunc.c bdfactor.c 
                     sparse.c sprow.c sparseio.c spchfctr.c 
                     splufctr.c spbkp.c spswap.c iter0.c 
                     itersym.c iternsym.c zmachine.c zcopy.c 
                     zmatio.c zmemory.c zvecop.c zmatop.c 
                     znorm.c zfunc.c zlufctr.c zsolve.c 
                     zmatlab.c zhsehldr.c zqrfctr.c zgivens.c 
                     zhessen.c zschur.c)
SET(MESCHACH_LIB "") 
FOREACH(MESCH_TARGET ${MESCHACH_TARGETS})
    LIST(APPEND MESCHACH_LIB "${CMAKE_CURRENT_LIST_DIR}/meschach_lib/${MESCH_TARGET}")
ENDFOREACH()

add_library(meschach_omp STATIC ${MESCHACH_LIB})

FOREACH(MESCH_TARGET ${MESCHACH_LIB})
    set_source_files_properties("${MESCH_TARGET}" PROPERTIES LANGUAGE CXX)
    set_source_files_properties("${MESCH_TARGET}" PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/meschach_lib")
ENDFOREACH()

set_target_properties(meschach_omp PROPERTIES LANGUAGE CXX)
target_compile_options(meschach_omp BEFORE PUBLIC "-DHAVE_CONFIG_H" "-DANSI_C")
target_link_options(meschach_omp BEFORE PRIVATE "-lm")

add_hecbench_benchmark(
    NAME leukocyte
    MODEL omp
    SOURCES detect_main.c
            find_ellipse.c
            track_ellipse.c
            track_ellipse_gpu.c
            avilib.c
            misc_math.c
    INCLUDE_DIRS meschach_lib
    CATEGORIES algorithms
    LINK_LIBRARIES meschach_omp 
    C_AS_CXX
)
