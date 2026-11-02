/* LUT-GEMM
 * Copyright (c) 2024-present NAVER Cloud Corp. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sycl/sycl.hpp>
#include "tests.h"

template<int M, int N, int K, int NUM_BITS, int A_GROUP_SIZE=K>
class int3_col_wise_matmul_fp16{
public:
    static const int num_groups = K/A_GROUP_SIZE;

    // float     qW[K   ][NUM_BITS][N]; // (-1, 1) 
    // uint32_t  bW[K/32][NUM_BITS][N]; // bit packed
    // float     alpha[num_groups][NUM_BITS][N];
    // float    q_bias[num_groups][N];
    float*     qW;
    uint32_t*  bW;
    float*     alpha;
    float*    q_bias;

    float*   weight;//[K][N];
    float*   input; //[M][K];

    sycl::half *d_weight_fp16;
    sycl::half *d_input;
    sycl::half *d_cu_output; // BLAS result
    sycl::half *d_nq_output;

    nQWeight_fp16 *nqW;

    sycl::queue q;

    // constructor
    int3_col_wise_matmul_fp16(sycl::queue &m_q) : q (m_q) {
       nqW = new nQWeight_fp16 (m_q);
    }

    ~int3_col_wise_matmul_fp16() {
      delete(nqW);
    }

    void run(int iter = 100) {
        alloc_memory();
        makeRandomInput();
        makeRandomWeight();
        makeRandomAlpha();
        dequantizeFrom_qW();
        fhCpy(d_input, input  ,M * K);
        fhCpy(d_weight_fp16, weight ,K * N);
        q.wait();

        nqW->parsing(bW, alpha, K, N, NUM_BITS, false, num_groups, q_bias);
        q.wait();

        // validate against BLAS GEMM
        double meanError = checkErr();
        printf("mean Error: %lf\n", meanError);

        lutgemm_latency(*nqW, M, N, K, d_input, d_weight_fp16, d_cu_output, iter);

        free_memory();
    }

    void lutgemm_latency(nQWeight_fp16 &nqW, int m, int n, int k, sycl::half *A,
                         sycl::half *B, sycl::half *C, int iter = 64) {
        timer tm;
        // warmup 
        matmul(q, (void*)C, (void*)A, nqW, m);
        q.wait();

        for(int i=0;i<iter;i++){
            tm.start();
            matmul(q, (void*)C, (void*)A, nqW, m);
            q.wait();
            tm.end();
        }
        printf("latency min : %.5fms, max : %.5fms, avg:%.5f\n", tm.min(), tm.max(), tm.mean());
    }

    double checkErr(){
        cublas_gemm_ex(q, d_input, d_weight_fp16, d_cu_output, M, N, K);
        q.memset(d_nq_output, 0, sizeof(sycl::half) * M * N).wait();
        matmul(q, d_nq_output, d_input, *nqW, M);
        q.wait();
        return checkOutputMeanError(d_cu_output, d_nq_output);
    }

    double checkOutputMeanError(sycl::half *o1, sycl::half *o2) {
        double err=0;
        for(int m=0;m<M;m++){
            for(int n=0;n<N;n++){
                err += std::abs(float(o1[m*N + n]) - float(o2[m*N + n]));
            }
        }
        return err/(M*N);
    }

    void makeRandomInput(){
        random_seed(); 
        for(int m=0;m<M;m++)
            for(int k=0;k<K;k++)
                input[m*K+k] = rand_fp32(); // (-1.0, 1.0) / 2^b
    }

    void makeRandomAlpha(){
        random_seed(); 
        for(int g=0;g<num_groups;g++)
            for(int n=0;n<N;n++){
                q_bias[g * N + n] = rand_fp32()/(1<< NUM_BITS);
                for(int b=0;b<NUM_BITS;b++)
                    alpha[g * (NUM_BITS*N) + b * N + n] = rand_fp32()/(1<<b); // (-1.0, 1.0) / 2^b
            }
    }

    void makeRandomWeight(){
        random_seed(); 
        for(int n=0;n<N;n++){
            for(int b=0;b<NUM_BITS;b++){
                for(int k=0;k<K;k+=32){
                    uint32_t s=0;
                    for(int t=0;t<32;t++){
                        if(rand_bool()){
                                s |= 1<<t;
                                qW[(k + t) * (NUM_BITS * N) + b * N + n] = +1;
                        } else  qW[(k + t) * (NUM_BITS * N) + b * N + n] = -1;
                    }
                    bW[k/32 * (NUM_BITS * N) + b * N + n] = s;
                }
            }
        }
    }

    void dequantizeFrom_qW(){
        for(int n=0;n<N;n++){
            for(int k=0;k<K;k++){
                weight[k*N+n] = q_bias[k/A_GROUP_SIZE * N + n];
                for(int b=0;b<NUM_BITS;b++){
                    weight[k*N+n] += alpha[k/A_GROUP_SIZE * (NUM_BITS*N) + b * N + n] * 
                                     qW[k * (NUM_BITS*N) + b * N + n]; 
                }
            }
        }        
    }

    void alloc_memory() {
        bW = (uint32_t*) malloc (sizeof(uint32_t) * K/32 * NUM_BITS * N);
        qW = (float*) calloc ((uint64_t)K * NUM_BITS * N, sizeof(float));
        alpha = (float*) malloc (sizeof(float) * num_groups * NUM_BITS * N);
        q_bias = (float*) malloc (sizeof(float) * num_groups * N);
        weight = (float*) malloc (sizeof(float) * K * N);
        input = (float*) malloc (sizeof(float) * M * K);
        d_input = (sycl::half *)sycl::malloc_shared(sizeof(sycl::half) * M * K,
                                                    q);
        d_weight_fp16 = (sycl::half *)sycl::malloc_shared(
            sizeof(sycl::half) * K * N, q);
        d_cu_output = (sycl::half *)sycl::malloc_shared(
            sizeof(sycl::half) * M * N, q);
        d_nq_output = (sycl::half *)sycl::malloc_shared(
            sizeof(sycl::half) * M * N, q);
    }

    void free_memory() {
        free(qW);
        free(bW);
        free(alpha);
        free(q_bias);
        free(weight);
        free(input);
        sycl::free(d_input, q);
        sycl::free(d_weight_fp16, q);
        sycl::free(d_cu_output, q);
        sycl::free(d_nq_output, q);
    }

    void fhCpy(sycl::half *a, float *b, int size) {
       for (int i = 0; i < size; i++)
           a[i] = sycl::vec<float, 1>(b[i])
                      .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    }

};

template <int H>
void test_case(const int repeat) {
#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    printf("M = 1, N = %d, K = %d\n", 4*H, H);

    printf("LUT-GEMM [INT8, FP16, FP16]\t");
    int3_col_wise_matmul_fp16<1, H*4, H, 8, 128> t_i8_f16_f16 {q};
    t_i8_f16_f16.run(repeat);

    printf("LUT-GEMM [INT4, FP16, FP16]\t");
    int3_col_wise_matmul_fp16<1, H*4, H, 4, 128> t_i4_f16_f16 {q};
    t_i4_f16_f16.run(repeat);

    printf("LUT-GEMM [INT3, FP16, FP16]\t");
    int3_col_wise_matmul_fp16<1, H*4, H, 3, 128> t_i3_f16_f16 {q};
    t_i3_f16_f16.run(repeat);
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
      printf("Usage: %s <test case> <repeat>\n", argv[0]);
      printf("Case 0: K = 1024\n");
      printf("Case 1: K = 4096\n");
      printf("Default: K = 12288\n");
      return 1;
    }

    const int option = atoi(argv[1]);
    const int repeat = atoi(argv[2]);

    if (option == 0)
      test_case<1024>(repeat);
    else if (option == 1)
      test_case<4096>(repeat);
    else
      test_case<12288>(repeat);

    return 0;
}
