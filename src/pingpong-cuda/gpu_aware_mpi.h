#ifndef GPU_AWARE_MPI_H
#define GPU_AWARE_MPI_H

// Decide, before any device pointer reaches MPI, whether the MPI library can
// take GPU buffers. Header-only; call gpu_aware_mpi_require() after MPI_Init.
//
// Each MPI is asked the way it documents:
//   Cray MPICH    MPICH_GPU_SUPPORT_ENABLED=1 (default off), then the MPICH
//                 query below when the headers have it.
//   Intel MPI     I_MPI_OFFLOAD nonzero (default 0). Intel and NVIDIA GPUs
//                 only; always no for AMD GPUs.
//   MVAPICH2      MV2_USE_CUDA / MV2_USE_ROCM (unset: cannot tell).
//   MPICH >= 4.0 (and derivatives)
//                 MPIX_Query_cuda_support / _hip_support / _ze_support. They
//                 honor MPIR_CVAR_ENABLE_GPU.
//   Open MPI      MPIX_Query_cuda_support (CUDA extension in mpi-ext.h) and
//                 MPIX_Query_rocm_support (ROCm extension, Open MPI 5).
//
// A "no" always aborts. When the library cannot tell (for example Open MPI
// 4.1 + UCX on ROCm, or an MPI not listed above), the program also aborts
// unless MPI_GPU_AWARE=1 is set to assert that the MPI is GPU-aware.
//
// MPIX_GPU_SUPPORT_CUDA / _ZE / _HIP in MPICH are type ids (0, 1, 2), not
// feature flags, so they cannot guard the calls. Gate on the MPICH version
// that added all three query functions instead.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(OPEN_MPI) && OPEN_MPI
#  if defined(__has_include)
#    if __has_include(<mpi-ext.h>)
#      include <mpi-ext.h>
#    endif
#  else
#    include <mpi-ext.h>
#  endif
#endif

#define GPU_AWARE_MPI_KIND_UNKNOWN 0
#define GPU_AWARE_MPI_KIND_CUDA 1
#define GPU_AWARE_MPI_KIND_HIP 2
#define GPU_AWARE_MPI_KIND_ZE 3

#define GPU_AWARE_MPI_NO 0
#define GPU_AWARE_MPI_YES 1
#define GPU_AWARE_MPI_UNKNOWN (-1)

// MPICH 4.0 in MPICH_NUMVERSION (major*1e7 + minor*1e5 + rev*1e3 +
// release_type*100 + patch, with a regular release type of 3).
#define GPU_AWARE_MPI_MPICH_QUERY_VERSION 40000300

#if defined(MPICH_NUMVERSION) && (MPICH_NUMVERSION >= GPU_AWARE_MPI_MPICH_QUERY_VERSION)
#  define GPU_AWARE_MPI_HAVE_MPICH_QUERY 1
#endif

typedef struct {
  int answer;
  const char *why;   // what decided the answer
  const char *hint;  // how to turn GPU support on, if known
} gpu_aware_mpi_result_t;

static inline const char *gpu_aware_mpi_kind_name(int kind)
{
  if (kind == GPU_AWARE_MPI_KIND_CUDA)
    return "CUDA";
  if (kind == GPU_AWARE_MPI_KIND_HIP)
    return "HIP";
  if (kind == GPU_AWARE_MPI_KIND_ZE)
    return "Level Zero";
  return "unknown-vendor";
}

// Map a device vendor string (for example SYCL's info::device::vendor) to a
// GPU kind. Matching is case-insensitive.
static inline int gpu_aware_mpi_kind_from_vendor(const char *vendor)
{
  char lower[256];
  size_t i = 0;
  for (; vendor != NULL && vendor[i] != '\0' && i + 1 < sizeof(lower); i++) {
    const char c = vendor[i];
    lower[i] = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
  }
  lower[i] = '\0';
  if (strstr(lower, "nvidia") != NULL)
    return GPU_AWARE_MPI_KIND_CUDA;
  if (strstr(lower, "amd") != NULL || strstr(lower, "advanced micro") != NULL)
    return GPU_AWARE_MPI_KIND_HIP;
  if (strstr(lower, "intel") != NULL)
    return GPU_AWARE_MPI_KIND_ZE;
  return GPU_AWARE_MPI_KIND_UNKNOWN;
}

static inline int gpu_aware_mpi_library_version_has(const char *needle)
{
  char version[MPI_MAX_LIBRARY_VERSION_STRING];
  int length = 0;
  if (MPI_Get_library_version(version, &length) != MPI_SUCCESS)
    return 0;
  if (length < 0)
    length = 0;
  if (length >= MPI_MAX_LIBRARY_VERSION_STRING)
    length = MPI_MAX_LIBRARY_VERSION_STRING - 1;
  version[length] = '\0';
  return strstr(version, needle) != NULL;
}

// 1 when the variable is set to a nonzero integer, 0 when set to zero,
// -1 when unset or empty.
static inline int gpu_aware_mpi_env_flag(const char *name)
{
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0')
    return -1;
  return atoi(value) != 0;
}

static inline gpu_aware_mpi_result_t gpu_aware_mpi_result(int answer, const char *why,
                                                   const char *hint)
{
  gpu_aware_mpi_result_t r;
  r.answer = answer;
  r.why = why;
  r.hint = hint;
  return r;
}

#if defined(GPU_AWARE_MPI_HAVE_MPICH_QUERY)
static inline gpu_aware_mpi_result_t gpu_aware_mpi_mpich_query(int kind, const char *hint)
{
  if (kind == GPU_AWARE_MPI_KIND_CUDA)
    return MPIX_Query_cuda_support() == 1
               ? gpu_aware_mpi_result(GPU_AWARE_MPI_YES, "MPIX_Query_cuda_support() returned 1", hint)
               : gpu_aware_mpi_result(GPU_AWARE_MPI_NO, "MPIX_Query_cuda_support() returned 0", hint);
  if (kind == GPU_AWARE_MPI_KIND_HIP)
    return MPIX_Query_hip_support() == 1
               ? gpu_aware_mpi_result(GPU_AWARE_MPI_YES, "MPIX_Query_hip_support() returned 1", hint)
               : gpu_aware_mpi_result(GPU_AWARE_MPI_NO, "MPIX_Query_hip_support() returned 0", hint);
  if (kind == GPU_AWARE_MPI_KIND_ZE)
    return MPIX_Query_ze_support() == 1
               ? gpu_aware_mpi_result(GPU_AWARE_MPI_YES, "MPIX_Query_ze_support() returned 1", hint)
               : gpu_aware_mpi_result(GPU_AWARE_MPI_NO, "MPIX_Query_ze_support() returned 0", hint);
  return gpu_aware_mpi_result(GPU_AWARE_MPI_UNKNOWN, "MPICH has no query for this GPU vendor", NULL);
}
#endif

static inline gpu_aware_mpi_result_t gpu_aware_mpi_cray_mpich(int kind)
{
  const char *hint = "set MPICH_GPU_SUPPORT_ENABLED=1 and link the Cray GTL library "
                     "(libmpi_gtl_cuda or libmpi_gtl_hsa)";
  if (gpu_aware_mpi_env_flag("MPICH_GPU_SUPPORT_ENABLED") != 1)
    return gpu_aware_mpi_result(GPU_AWARE_MPI_NO,
                           "Cray MPICH with MPICH_GPU_SUPPORT_ENABLED unset or 0", hint);
#if defined(GPU_AWARE_MPI_HAVE_MPICH_QUERY)
  return gpu_aware_mpi_mpich_query(kind, hint);
#else
  (void)kind;
  return gpu_aware_mpi_result(GPU_AWARE_MPI_YES, "Cray MPICH with MPICH_GPU_SUPPORT_ENABLED=1", hint);
#endif
}

static inline gpu_aware_mpi_result_t gpu_aware_mpi_intel_mpi(int kind)
{
  if (kind == GPU_AWARE_MPI_KIND_HIP)
    return gpu_aware_mpi_result(GPU_AWARE_MPI_NO,
                           "Intel MPI supports Intel and NVIDIA GPU buffers, not AMD",
                           "use a ROCm-aware MPI (Cray MPICH with libmpi_gtl_hsa, "
                           "Open MPI with ROCm, or MPICH built with HIP)");
  const char *hint = "set I_MPI_OFFLOAD=1";
  if (gpu_aware_mpi_env_flag("I_MPI_OFFLOAD") == 1)
    return gpu_aware_mpi_result(GPU_AWARE_MPI_YES, "Intel MPI with I_MPI_OFFLOAD nonzero", hint);
  return gpu_aware_mpi_result(GPU_AWARE_MPI_NO, "Intel MPI with I_MPI_OFFLOAD unset or 0", hint);
}

static inline gpu_aware_mpi_result_t gpu_aware_mpi_mvapich2(int kind)
{
  const char *name = NULL;
  const char *hint = NULL;
  if (kind == GPU_AWARE_MPI_KIND_CUDA) {
    name = "MV2_USE_CUDA";
    hint = "set MV2_USE_CUDA=1 (MVAPICH2-GDR or a CUDA-enabled MVAPICH2 build)";
  } else if (kind == GPU_AWARE_MPI_KIND_HIP) {
    name = "MV2_USE_ROCM";
    hint = "set MV2_USE_ROCM=1 (MVAPICH2-GDR built with ROCm)";
  } else {
    return gpu_aware_mpi_result(GPU_AWARE_MPI_UNKNOWN,
                           "MVAPICH2 has no switch for this GPU vendor", NULL);
  }
  const int flag = gpu_aware_mpi_env_flag(name);
  if (flag == 1)
    return gpu_aware_mpi_result(GPU_AWARE_MPI_YES,
                           kind == GPU_AWARE_MPI_KIND_CUDA ? "MVAPICH2 with MV2_USE_CUDA=1"
                                                          : "MVAPICH2 with MV2_USE_ROCM=1",
                           hint);
  if (flag == 0)
    return gpu_aware_mpi_result(GPU_AWARE_MPI_NO,
                           kind == GPU_AWARE_MPI_KIND_CUDA ? "MVAPICH2 with MV2_USE_CUDA=0"
                                                          : "MVAPICH2 with MV2_USE_ROCM=0",
                           hint);
  return gpu_aware_mpi_result(GPU_AWARE_MPI_UNKNOWN,
                         kind == GPU_AWARE_MPI_KIND_CUDA ? "MVAPICH2 with MV2_USE_CUDA unset"
                                                        : "MVAPICH2 with MV2_USE_ROCM unset",
                         hint);
}

#if defined(OPEN_MPI) && OPEN_MPI
static inline gpu_aware_mpi_result_t gpu_aware_mpi_openmpi(int kind)
{
  if (kind == GPU_AWARE_MPI_KIND_CUDA) {
    const char *hint = "use an Open MPI built with CUDA support (for example the NVIDIA HPC-X or HPC SDK MPI)";
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA || defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_Query_cuda_support() == 1
               ? gpu_aware_mpi_result(GPU_AWARE_MPI_YES, "MPIX_Query_cuda_support() returned 1", hint)
               : gpu_aware_mpi_result(GPU_AWARE_MPI_NO, "MPIX_Query_cuda_support() returned 0", hint);
#else
    return gpu_aware_mpi_result(GPU_AWARE_MPI_UNKNOWN,
                           "Open MPI headers have no CUDA extension (mpi-ext.h)", hint);
#endif
  }
  if (kind == GPU_AWARE_MPI_KIND_HIP) {
    const char *hint = "use an Open MPI built with ROCm support (Open MPI 5 reports it; "
                       "Open MPI 4 with ROCm-enabled UCX cannot report it)";
#if defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM || defined(MPIX_ROCM_AWARE_SUPPORT)
    return MPIX_Query_rocm_support() == 1
               ? gpu_aware_mpi_result(GPU_AWARE_MPI_YES, "MPIX_Query_rocm_support() returned 1", hint)
               : gpu_aware_mpi_result(GPU_AWARE_MPI_NO, "MPIX_Query_rocm_support() returned 0", hint);
#else
    return gpu_aware_mpi_result(GPU_AWARE_MPI_UNKNOWN,
                           "Open MPI headers have no ROCm extension (mpi-ext.h, Open MPI 5+)", hint);
#endif
  }
  return gpu_aware_mpi_result(GPU_AWARE_MPI_UNKNOWN,
                         "Open MPI has no query for this GPU vendor", NULL);
}
#endif

// Call after MPI_Init. Does not abort.
static inline gpu_aware_mpi_result_t gpu_aware_mpi_query(int kind)
{
  if (kind == GPU_AWARE_MPI_KIND_UNKNOWN)
    return gpu_aware_mpi_result(GPU_AWARE_MPI_UNKNOWN, "the GPU vendor is not recognized", NULL);

  // Runtime library checks first: these MPIs are MPICH-derived, and their
  // own switch decides even when the MPICH query is in the headers.
  if (gpu_aware_mpi_library_version_has("CRAY MPICH"))
    return gpu_aware_mpi_cray_mpich(kind);
  if (gpu_aware_mpi_library_version_has("Intel(R) MPI") ||
      gpu_aware_mpi_library_version_has("Intel MPI"))
    return gpu_aware_mpi_intel_mpi(kind);
  if (gpu_aware_mpi_library_version_has("MVAPICH2"))
    return gpu_aware_mpi_mvapich2(kind);

#if defined(OPEN_MPI) && OPEN_MPI
  return gpu_aware_mpi_openmpi(kind);
#elif defined(GPU_AWARE_MPI_HAVE_MPICH_QUERY)
  return gpu_aware_mpi_mpich_query(kind, "enable GPU support in MPICH (MPIR_CVAR_ENABLE_GPU=1, "
                                    "and a build configured with CUDA, HIP, or Level Zero)");
#else
  return gpu_aware_mpi_result(GPU_AWARE_MPI_UNKNOWN,
                         "this MPI has no GPU-buffer query (not Cray MPICH, Intel MPI, "
                         "MVAPICH2, Open MPI, or MPICH >= 4.0)", NULL);
#endif
}

// Abort unless the MPI library reports GPU-buffer support, or it cannot tell
// and MPI_GPU_AWARE=1 is set.
static inline void gpu_aware_mpi_require(int kind, int rank)
{
  const gpu_aware_mpi_result_t r = gpu_aware_mpi_query(kind);
  const char *kind_name = gpu_aware_mpi_kind_name(kind);

  if (r.answer == GPU_AWARE_MPI_YES)
    return;

  if (r.answer == GPU_AWARE_MPI_UNKNOWN && gpu_aware_mpi_env_flag("MPI_GPU_AWARE") == 1) {
    if (rank == 0) {
      printf("WARNING: MPI cannot report %s GPU-buffer support (%s); "
             "continuing because MPI_GPU_AWARE=1.\n", kind_name, r.why);
      fflush(stdout);
    }
    return;
  }

  if (r.answer == GPU_AWARE_MPI_NO) {
    fprintf(stderr,
            "ERROR: rank %d: MPI library reports no %s GPU-buffer support (%s).\n",
            rank, kind_name, r.why);
  } else {
    fprintf(stderr,
            "ERROR: rank %d: MPI library cannot report %s GPU-buffer support (%s).\n"
            "If this MPI is GPU-aware, set MPI_GPU_AWARE=1 to continue.\n",
            rank, kind_name, r.why);
  }
  if (r.hint)
    fprintf(stderr, "To enable GPU buffers: %s.\n", r.hint);
  fprintf(stderr, "This program passes GPU device pointers to MPI.\n");
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
  // MPI_Abort may return; never fall through to passing device pointers.
  _exit(1);
}

#endif
