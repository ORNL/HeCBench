/*
def hadamard_transform_ref(x, scale=1.0):
    """
    x: (..., dim)
    out: (..., dim)
    """
    if hadamard is None:
        raise ImportError("Please install scipy")
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * scale
    return out[..., :dim].reshape(*x_shape)
*/

inline int next_power_of_two(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

void hadamard_transform(std::vector<float>& data, int dim) {
    // dim must be power of 2
    for (int len = 1; len < dim; len <<= 1) {
        for (int i = 0; i < dim; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j] = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
}

template <typename T>
void reference (
    std::vector<T> &x,
    std::vector<T> &output,
    int batch, // number of rows after flattening
    int dim,
    float scale
)
{
    assert(x.size() == (int64_t)batch * dim);

    int dim_padded = next_power_of_two(dim);

    for (int b = 0; b < batch; ++b) {
        // Copy + pad
        std::vector<float> temp(dim_padded, 0.0f);
        for (int i = 0; i < dim; ++i) {
            temp[i] = x[b * dim + i];
        }

        // Hadamard transform
        hadamard_transform(temp, dim_padded);

        // Scale and truncate
        for (int i = 0; i < dim; ++i) {
            output[b * dim + i] = temp[i] * scale;
        }
    }
}

// eps is the machine epsilon of T; both outputs are rounded to T from fp32,
// so they may differ by about one ULP of T.
template <typename T>
bool verify (
    const std::vector<T> &output,
    const std::vector<T> &ref_output,
    float eps
)
{
    const float rel_tol = 2.0f * eps;
    const float abs_tol = 1e-5f;

    for (size_t i = 0; i < output.size(); ++i) {
        float out = (float)output[i];
        float ref = (float)ref_output[i];
        float diff = std::fabs(out - ref);
        float bound = rel_tol * std::fabs(ref) + abs_tol;
        if (!std::isfinite(out) || diff > bound) {
            printf("Mismatch at index %zu %f %f (diff=%e, bound=%e)\n",
                   i, out, ref, diff, bound);
            return false;
        }
    }
    return true;
}
