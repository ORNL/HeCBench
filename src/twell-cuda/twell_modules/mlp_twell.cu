#include "torch_compat.h"
#include <cuda_runtime.h>

#include <cstdint>

namespace TWELL_D2T {
void run_d2t_layer_128x256x64TS8(
    const int layer_number,
    at::BFloat16* A_d,
    uint32_t* C_packed_d,
    cudaStream_t stream = 0
);
}

namespace TWELL_T2D {
void mm_t2d_wid(
    uint32_t* in_packed,
    at::BFloat16* down_weight,
    at::BFloat16* out,
    const size_t IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM,
    const int NUM_SPLITS
);
}

namespace TWELL_GATED_T2D {
void mm_t2d_wid(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    at::BFloat16* out,
    const size_t IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM
);

void mm_t2d_wid_inplace(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    const size_t IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM
);

void mm_t2d_wid_high_precision(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    at::BFloat16* out,
    const size_t IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM
);

void mm_t2d_wid_high_precision_inplace(
    at::BFloat16* in_dense,
    uint32_t* gate_out_twell_packed,
    at::BFloat16* up_weight,
    at::BFloat16* down_weight,
    const size_t IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM
);
}

namespace TWELL_MLP {

void run_mlp_layer_128x256x64TS8(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    at::BFloat16* out_d,
    const size_t M,
    const int FEATURE_DIM,
    const int OUT_DIM,
    const int NUM_SPLITS
) {
    if (M == 0 || FEATURE_DIM <= 0 || OUT_DIM <= 0) {
        return;
    }

    TWELL_D2T::run_d2t_layer_128x256x64TS8(layer_number, A_d, C_packed_d);
    TWELL_T2D::mm_t2d_wid(
        C_packed_d,
        down_weight_d,
        out_d,
        M,
        FEATURE_DIM,
        OUT_DIM,
        NUM_SPLITS
    );
}

void run_mlp_layer_128x256x64TS8_inplace(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    const size_t M,
    const int FEATURE_DIM,
    const int OUT_DIM,
    const int NUM_SPLITS
) {
    if (M == 0 || FEATURE_DIM <= 0 || OUT_DIM <= 0) {
        return;
    }

    TWELL_D2T::run_d2t_layer_128x256x64TS8(layer_number, A_d, C_packed_d);
    TWELL_T2D::mm_t2d_wid(
        C_packed_d,
        down_weight_d,
        A_d,
        M,
        FEATURE_DIM,
        OUT_DIM,
        NUM_SPLITS
    );
}

void run_gated_mlp_layer_128x256x64TS8(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* up_weight_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    at::BFloat16* out_d,
    const size_t M,
    const int FEATURE_DIM,
    const int OUT_DIM
) {
    if (M == 0 || FEATURE_DIM <= 0 || OUT_DIM <= 0) {
        return;
    }

    TWELL_D2T::run_d2t_layer_128x256x64TS8(layer_number, A_d, C_packed_d);
    TWELL_GATED_T2D::mm_t2d_wid(
        A_d,
        C_packed_d,
        up_weight_d,
        down_weight_d,
        out_d,
        M,
        FEATURE_DIM,
        OUT_DIM
    );
}

void run_gated_mlp_layer_128x256x64TS8_high_precision(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* up_weight_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    at::BFloat16* out_d,
    const size_t M,
    const int FEATURE_DIM,
    const int OUT_DIM
) {
    if (M == 0 || FEATURE_DIM <= 0 || OUT_DIM <= 0) {
        return;
    }

    TWELL_D2T::run_d2t_layer_128x256x64TS8(layer_number, A_d, C_packed_d);
    TWELL_GATED_T2D::mm_t2d_wid_high_precision(
        A_d,
        C_packed_d,
        up_weight_d,
        down_weight_d,
        out_d,
        M,
        FEATURE_DIM,
        OUT_DIM
    );
}

void run_gated_mlp_layer_128x256x64TS8_inplace(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* up_weight_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    const size_t M,
    const int FEATURE_DIM,
    const int OUT_DIM
) {
    if (M == 0 || FEATURE_DIM <= 0 || OUT_DIM <= 0) {
        return;
    }

    TWELL_D2T::run_d2t_layer_128x256x64TS8(layer_number, A_d, C_packed_d);
    TWELL_GATED_T2D::mm_t2d_wid_inplace(
        A_d,
        C_packed_d,
        up_weight_d,
        down_weight_d,
        M,
        FEATURE_DIM,
        OUT_DIM
    );
}

void run_gated_mlp_layer_128x256x64TS8_high_precision_inplace(
    const int layer_number,
    at::BFloat16* A_d,
    at::BFloat16* up_weight_d,
    at::BFloat16* down_weight_d,
    uint32_t* C_packed_d,
    const size_t M,
    const int FEATURE_DIM,
    const int OUT_DIM
) {
    if (M == 0 || FEATURE_DIM <= 0 || OUT_DIM <= 0) {
        return;
    }

    TWELL_D2T::run_d2t_layer_128x256x64TS8(layer_number, A_d, C_packed_d);
    TWELL_GATED_T2D::mm_t2d_wid_high_precision_inplace(
        A_d,
        C_packed_d,
        up_weight_d,
        down_weight_d,
        M,
        FEATURE_DIM,
        OUT_DIM
    );
}

}  // namespace TWELL_MLP
