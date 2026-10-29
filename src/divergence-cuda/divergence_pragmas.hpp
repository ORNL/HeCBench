#ifndef DIVERGENCE_PRAGMAS_HPP
#define DIVERGENCE_PRAGMAS_HPP

#if defined(__INTEL_COMPILER)
#define ALIGN(vardec) __declspec(align) vardec
#define ALIGNTO(vardec, boundary) \
  __declspec(align(boundary)) vardec
#elif defined(__clang__) || defined(__GNUG__)
#define ALIGN(vardec) __attribute__((aligned)) vardec
#define ALIGNTO(vardec, boundary) \
  __attribute__((aligned(boundary))) vardec
#endif

#endif
