// to single-precision
#define ToF(input) static_cast<accscalar_t>(input)
// to half-precision
#define ToH(input) static_cast<scalar_t>(input)

template<typename T>
T sigmoidf(T in)  {
  return 1.f / (1.f + expf(-in));
}

template <typename scalar_t, typename accscalar_t, typename index_type>
void reference(
    scalar_t* Input,
    scalar_t* Hidden,
    scalar_t* Bias1,
    scalar_t* Bias2,
    scalar_t* hx,
    scalar_t* hy,
    scalar_t* storage,
    index_type hsz,
    index_type totalElements)
{
  for (index_type linearIndex = 0; linearIndex < totalElements; linearIndex++) {

    // ------------------------------------------------------------------
    // Input / Hidden gate reads
    // ------------------------------------------------------------------
    index_type batch_idx  = linearIndex / hsz;
    index_type hidden_idx = linearIndex % hsz;
    index_type inp_offset = batch_idx * 3 * hsz + hidden_idx;

    scalar_t ir = Input [inp_offset + 0 * hsz];   // reset  input
    scalar_t ii = Input [inp_offset + 1 * hsz];   // update input
    scalar_t in_ = Input[inp_offset + 2 * hsz];   // new    input

    scalar_t hr = Hidden[inp_offset + 0 * hsz];   // reset  hidden
    scalar_t hi = Hidden[inp_offset + 1 * hsz];   // update hidden
    scalar_t hn = Hidden[inp_offset + 2 * hsz];   // new    hidden

    scalar_t hx_ = hx[linearIndex];               // h(t-1)

    // ------------------------------------------------------------------
    // Bias reads
    // Bias1/Bias2 are [3, hsz]: same layout as one gate-row of Input
    // ------------------------------------------------------------------
    scalar_t b1r = Bias1[hidden_idx + 0 * hsz];
    scalar_t b1i = Bias1[hidden_idx + 1 * hsz];
    scalar_t b1n = Bias1[hidden_idx + 2 * hsz];

    scalar_t b2r = Bias2[hidden_idx + 0 * hsz];
    scalar_t b2i = Bias2[hidden_idx + 1 * hsz];
    scalar_t b2n = Bias2[hidden_idx + 2 * hsz];

    // ------------------------------------------------------------------
    // GRU gate computations
    // ------------------------------------------------------------------

    // Reset gate r = sigmoidf(ir + hr + b1r + b2r)
    accscalar_t rg = sigmoidf(ToF(ir) + ToF(hr) + ToF(b1r) + ToF(b2r));

    // Update gate z = sigmoidf(ii + hi + b1i + b2i)
    accscalar_t ig = sigmoidf(ToF(ii) + ToF(hi) + ToF(b1i) + ToF(b2i));

    // New (candidate) gate n = tanh(in + b1n + r*(hn + b2n))
    accscalar_t ng = ToF(in_) + ToF(b1n) + rg * (ToF(hn) + ToF(b2n));
    ng = std::tanh(ng);

    // Output h(t) = n + z*(h(t-1) - n)
    hy[linearIndex] = ToH(ng + ig * (ToF(hx_) - ng));

    // ------------------------------------------------------------------
    // Save intermediates for backward pass
    // ------------------------------------------------------------------
    index_type stor_offset = batch_idx * 5 * hsz + hidden_idx;

    storage[stor_offset + 0 * hsz] = ToH(rg);                 // reset gate
    storage[stor_offset + 1 * hsz] = ToH(ig);                 // update gate
    storage[stor_offset + 2 * hsz] = ToH(ng);                 // new gate
    storage[stor_offset + 3 * hsz] = hx_;                     // h(t-1)
    storage[stor_offset + 4 * hsz] = ToH(ToF(hn) + ToF(b2n)); // hn + b2n
  }
}
