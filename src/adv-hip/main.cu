#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define p_IJWID 6
#define p_JID   4
#define p_JWID  5
#define p_Np    512
#define p_Nq    8
#define p_Nvgeo 12
#define p_RXID  0
#define p_RYID  1
#define p_RZID  7
#define p_SXID  2
#define p_SYID  3
#define p_SZID  8
#define p_TXID  9
#define p_TYID  10
#define p_TZID  11
#define p_cubNp 4096
#define p_cubNq 16

// kernel
#include "adv.h"

#include "reference.h"

dfloat *drandAlloc(int N){
  dfloat *v = (dfloat*) malloc(N * sizeof(dfloat));
  for(int n = 0; n < N; ++n) v[n] = drand48();
  return v;
}

int main(int argc, char **argv) {

  if (argc < 4) {
    printf("Usage: ./adv N cubN numElements [nRepetitions]\n");
    exit(-1);
  }

  const int N = atoi(argv[1]);
  const int cubN = atoi(argv[2]);
  const dlong Nelements = atoi(argv[3]);
  int Ntests = 1;

  if(argc >= 5) Ntests = atoi(argv[4]);

  const int Nq = N+1;
  const int cubNq = cubN+1;
  const int Np = Nq*Nq*Nq;
  const int cubNp = cubNq*cubNq*cubNq;
  const dlong offset = Nelements*Np;

  printf("Data type in bytes: %zu\n", sizeof(dfloat));

  srand48(123);
  dfloat *vgeo           = drandAlloc(Np*Nelements*p_Nvgeo);
  dfloat *cubvgeo        = drandAlloc(cubNp*Nelements*p_Nvgeo);
  dfloat *cubDiffInterpT = drandAlloc(3*cubNp*Nelements);
  dfloat *cubInterpT     = drandAlloc(Np*cubNp);
  dfloat *u              = drandAlloc(3*Np*Nelements);
  dfloat *adv            = drandAlloc(3*Np*Nelements);
  dfloat *adv_ref        = drandAlloc(3*Np*Nelements);

  dfloat *d_vgeo, *d_cubvgeo, *d_cubDiffInterpT, *d_cubInterpT, *d_u, *d_adv;
  hipMalloc((void**)&d_vgeo, Np*Nelements*p_Nvgeo*sizeof(dfloat));
  hipMalloc((void**)&d_cubvgeo, cubNp*Nelements*p_Nvgeo*sizeof(dfloat));
  hipMalloc((void**)&d_cubDiffInterpT,3*cubNp*Nelements*sizeof(dfloat));
  hipMalloc((void**)&d_cubInterpT, Np*cubNp*sizeof(dfloat));
  hipMalloc((void**)&d_u, 3*Np*Nelements*sizeof(dfloat));
  hipMalloc((void**)&d_adv, 3*Np*Nelements*sizeof(dfloat));

  hipMemcpy(d_vgeo, vgeo, Np*Nelements*p_Nvgeo*sizeof(dfloat), hipMemcpyHostToDevice);
  hipMemcpy(d_cubvgeo, cubvgeo, cubNp*Nelements*p_Nvgeo*sizeof(dfloat), hipMemcpyHostToDevice);
  hipMemcpy(d_cubDiffInterpT, cubDiffInterpT, 3*cubNp*Nelements*sizeof(dfloat), hipMemcpyHostToDevice);
  hipMemcpy(d_cubInterpT, cubInterpT, Np*cubNp*sizeof(dfloat), hipMemcpyHostToDevice);
  hipMemcpy(d_u, u, 3*Np*Nelements*sizeof(dfloat), hipMemcpyHostToDevice);
  hipMemcpy(d_adv, adv, 3*Np*Nelements*sizeof(dfloat), hipMemcpyHostToDevice);

  hipDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();

  // run kernel
  for(int test=0;test<Ntests;++test)
    advCubatureHex3D<<<dim3(Nelements, 1), dim3(16, 16)>>>(
        Nelements,
        d_vgeo,
        d_cubvgeo,
        d_cubDiffInterpT,
        d_cubInterpT,
        offset,
        d_u,
        d_adv);

  hipDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  const double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / Ntests;

  hipMemcpy(adv, d_adv, 3*Np*Nelements*sizeof(dfloat), hipMemcpyDeviceToHost);

  hipFree(d_vgeo);
  hipFree(d_cubvgeo);
  hipFree(d_cubDiffInterpT);
  hipFree(d_cubInterpT);
  hipFree(d_u);
  hipFree(d_adv);

  reference(Nelements,
            vgeo,
            cubvgeo,
            cubDiffInterpT,
            cubInterpT,
            offset,
            u,
            adv_ref);

  bool ok = true;
  for (int i = 0; i < 3*Np*Nelements; i++) {
    if (fabs(adv[i] - adv_ref[i]) > 1e-4) {
      std::cout << i << " : " << adv[i] << " != " << adv_ref[i] << std::endl;
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  // statistics
  const dfloat GDOFPerSecond = (N*N*N)*Nelements/elapsed;
  std::cout << " NRepetitions=" << Ntests
            << " N=" << N
            << " cubN=" << cubN
            << " Nelements=" << Nelements
            << " elapsed time=" << elapsed
            << " GDOF/s=" << GDOFPerSecond
            << "\n";

  free(vgeo          );
  free(cubvgeo       );
  free(cubDiffInterpT);
  free(cubInterpT    );
  free(u             );
  free(adv           );
  free(adv_ref       );
  return 0;
}
