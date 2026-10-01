/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <functional>

#include <cublasLt.h>
#include <cuda_fp8.h>
//#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

// Encode a float into an 8-bit E4M3 code (NVIDIA uses the OCP-style E4M3).
inline uint8_t encodeE4M3(float v) {
    __nv_fp8_e4m3 t(v);
    return (uint8_t)t.__x;
}

// Decode an 8-bit E4M3 code back to float.
inline float decodeE4M3(uint8_t b) {
    __nv_fp8_e4m3 t(0.0f);
    t.__x = (__nv_fp8_storage_t)b;
    return float(t);
}

template <typename InType, typename CType, typename OutType, typename ComputeType>
struct TestBench {
    using SampleRunner = std::function<void()>;

    TestBench(int m, int n, int k,
            ComputeType alpha = ComputeType{0.0f}, ComputeType beta = ComputeType{0.0f},
            size_t workspaceSize = 1024 * 1024 * 4, int N = 1,
            ComputeType Ascale = ComputeType{1.0f}, ComputeType Bscale = ComputeType{1.0f},
            ComputeType Cscale = ComputeType{1.0f}, ComputeType Dscale = ComputeType{1.0f}) :
        m(m), n(n), k(k), N(N), alpha(alpha), beta(beta), workspaceSize(workspaceSize), 
        Ahost(m * k * N), Bhost(n * k * N), Chost(m * n * N), Dhost(m * n * N),
        AscaleHost(Ascale), BscaleHost(Bscale), CscaleHost(Cscale), DscaleHost(Dscale) {
        checkCublasStatus(cublasLtCreate(&ltHandle));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Adev), m * k * N * sizeof(InType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Bdev), n * k * N  * sizeof(InType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Cdev), m * n * N  * sizeof(CType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Ddev), m * n * N  * sizeof(OutType)));
        checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
        checkCudaStatus(cudaStreamCreate(&stream));

        // Currently only fp8 supports per-tensor scaling
        perTensorScalingEnabled = std::is_same<InType, __nv_fp8_e4m3>::value || std::is_same<InType, __nv_fp8_e5m2>::value;

        if (perTensorScalingEnabled) {
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&AscaleDev), sizeof(*AscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&BscaleDev), sizeof(*BscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&CscaleDev), sizeof(*CscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DscaleDev), sizeof(*DscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DamaxDev), sizeof(*DamaxDev)));
        }

        fillData();
    }

    ~TestBench() {
        checkCublasStatus(cublasLtDestroy(ltHandle));
        checkCudaStatus(cudaFree(Adev));
        checkCudaStatus(cudaFree(Bdev));
        checkCudaStatus(cudaFree(Cdev));
        checkCudaStatus(cudaFree(Ddev));
        checkCudaStatus(cudaFree(workspace));
        if (perTensorScalingEnabled) {
            checkCudaStatus(cudaFree(AscaleDev));
            checkCudaStatus(cudaFree(BscaleDev));
            checkCudaStatus(cudaFree(CscaleDev));
            checkCudaStatus(cudaFree(DscaleDev));
            checkCudaStatus(cudaFree(DamaxDev));
        }
        checkCudaStatus(cudaStreamDestroy(stream));
    }

    // Uniform value used for the FP8 A and B inputs so the exact GEMM result is
    // known. 1.0 is exactly representable in E4M3; with all scales = 1 and
    // beta = 0 every output element equals FILL_VALUE^2 * K, which the BF16
    // output represents exactly for every benchmark shape.
    static constexpr float FILL_VALUE = 1.0f;

    void fillData() {
        uint8_t aByte = encodeE4M3(FILL_VALUE);
        uint8_t *ap = reinterpret_cast<uint8_t*>(Ahost.data());
        uint8_t *bp = reinterpret_cast<uint8_t*>(Bhost.data());
        for (int i = 0; i < m * k * N; i++) ap[i] = aByte;
        for (int i = 0; i < n * k * N; i++) bp[i] = aByte;
        // beta = 0, so C is unused; keep it zeroed.
        for (int i = 0; i < m * n * N; i++) Chost[i] = CType(0.0f);
    }

    // CPU reference check. With uniform inputs, alpha = 1, beta = 0 and all
    // per-tensor input scales = 1, every output element equals FILL_VALUE^2 * K.
    // The output D is BF16.
    bool verify(double relTol = 1e-2) {
        double expected = (double)FILL_VALUE * (double)FILL_VALUE * (double)k;
        double maxErr = 0.0;
        for (int i = 0; i < m * n * N; i++)
            maxErr = std::max(maxErr, std::fabs((double)__bfloat162float(Dhost[i]) - expected));
        bool ok = maxErr <= relTol * (std::fabs(expected) + 1.0);
        printf("Verification: expected %.6g, max abs error %.4g -> %s\n",
               expected, maxErr, ok ? "PASS" : "FAIL");
        return ok;
    }

    void copyDataToDevice() {
        checkCudaStatus(cudaMemcpyAsync(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(Cdev, Chost.data(), Chost.size() * sizeof(Chost[0]), cudaMemcpyHostToDevice, stream));
        if (perTensorScalingEnabled) {
            checkCudaStatus(cudaMemcpyAsync(AscaleDev, &AscaleHost, sizeof(AscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(BscaleDev, &BscaleHost, sizeof(BscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(CscaleDev, &CscaleHost, sizeof(CscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(DscaleDev, &DscaleHost, sizeof(DscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(DamaxDev, &DamaxHost, sizeof(DamaxHost), cudaMemcpyHostToDevice));
        }
    }

    void copyDataFromDevice() {
        checkCudaStatus(cudaMemcpyAsync(Dhost.data(), Ddev, Dhost.size() * sizeof(Dhost[0]), cudaMemcpyDeviceToHost, stream));
    }

    void streamSynchronize() {
        checkCudaStatus(cudaStreamSynchronize(stream));
    }

    void run(const SampleRunner& runSample) {
        copyDataToDevice();

        runSample();

        copyDataFromDevice();
        streamSynchronize();
    }

    bool perTensorScalingEnabled;
    int m, n, k, N;
    ComputeType alpha, beta;
    size_t workspaceSize;
    std::vector<InType> Ahost, Bhost;
    std::vector<CType> Chost;
    std::vector<OutType> Dhost;
    void *workspace;
    InType *Adev, *Bdev;
    CType *Cdev;
    OutType *Ddev;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;
    ComputeType AscaleHost, BscaleHost, CscaleHost, DscaleHost, DamaxHost;
    ComputeType *AscaleDev, *BscaleDev, *CscaleDev, *DscaleDev, *DamaxDev;
};
