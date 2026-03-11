
static float LCG_random_ref(unsigned int *seed) {
    const unsigned int m = 2147483648u;
    const unsigned int a = 26757677u;
    const unsigned int c = 1u;
    *seed = (a * (*seed) + c) % m;
    return (float)(*seed) / (float)m;
}

// =============================================================================
// 2. init — fills data[0..size) with LCG pseudo-random values
//
// CUDA kernel: each thread handles one index, seed = index ^ size.
// C++ reference: loop over all indices, same seed formula per element.
// =============================================================================
static void init_ref(float *data, int size) {
    for (int index = 0; index < size; index++) {
        unsigned int seed = (unsigned int)index ^ (unsigned int)size;
        data[index] = LCG_random_ref(&seed);
    }
}

static void elementwise_ref(
    int hiddenSize, int miniBatch,
    const float *tmp_h,
    const float *tmp_i,
    const float *bias,
    float       *linearGates,
    float       *h_out,
    float       *i_out,
    const float *c_in,
    float       *c_out)
{
    int numElements = miniBatch * hiddenSize;

    for (int index = 0; index < numElements; index++) {

        int batch     = index / hiddenSize;
        int hiddenIdx = index % hiddenSize;
        int gateIndex = hiddenIdx + 4 * batch * hiddenSize;

        // Accumulate pre-activation gate values
        float g[4];
        for (int i = 0; i < 4; i++) {
            g[i] = tmp_i[i * hiddenSize + gateIndex]
                 + tmp_h[i * hiddenSize + gateIndex];
            g[i] += bias[ i      * hiddenSize + hiddenIdx]
                  + bias[(i + 4) * hiddenSize + hiddenIdx];
            linearGates[gateIndex + i * hiddenSize] = g[i];
        }

        // Gate activations
        float in_gate     = 1.f / (1.f + expf(-g[0]));   // input  gate  i
        float forget_gate = 1.f / (1.f + expf(-g[1]));   // forget gate  f
        float in_gate2    = tanhf(g[2]);                 // cell   gate  g~
        float out_gate    = 1.f / (1.f + expf(-g[3]));   // output gate  o

        // Cell and hidden state update
        float val  = (forget_gate * c_in[index]) + (in_gate * in_gate2);
        c_out[index] = val;

        val = out_gate * tanhf(val);
        h_out[index] = val;
        i_out[index] = val;
    }
}

// multi-layer, multi-timestep LSTM scheduler
static void test_ref(int hiddenSize, int miniBatch, int seqLength, int numLayers,
                     float *testOutputi, float *testOutputh, float *testOutputc)
{
    int numElements = hiddenSize * miniBatch;

    int hc_size    = (seqLength + 1) * numLayers  * numElements;
    int i_size     =  seqLength      * (numLayers + 1) * numElements;
    int bias_size  =  numLayers * hiddenSize * 8;
    int tmp_h_size =  4 * numLayers  * numElements;
    int tmp_i_size =  4 * seqLength  * numElements;
    int lg_size    =  4 * seqLength  * numLayers * numElements;

    float *h_data      = (float*)malloc(hc_size   * sizeof(float));
    float *i_data      = (float*)malloc(i_size    * sizeof(float));
    float *c_data      = (float*)malloc(hc_size   * sizeof(float));
    float *bias        = (float*)malloc(bias_size * sizeof(float));
    float *tmp_h       = (float*)malloc(tmp_h_size * sizeof(float));
    float *tmp_i       = (float*)malloc(tmp_i_size * sizeof(float));
    float *linearGates = (float*)malloc(lg_size   * sizeof(float));

    if (!h_data || !i_data || !c_data || !bias ||
        !tmp_h  || !tmp_i  || !linearGates) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    // Initialise with the same LCG as the CUDA init kernel
    // (only tmp_h, tmp_i, c_data, bias are initialised — same as CUDA)
    init_ref(tmp_h,  tmp_h_size);
    init_ref(tmp_i,  tmp_i_size);
    init_ref(c_data, hc_size);
    init_ref(bias,   bias_size);

    // h_data and i_data are not explicitly initialised in the CUDA version
    // (cudaMalloc does not zero). Zero here for determinism.
    memset(h_data, 0, hc_size   * sizeof(float));
    memset(i_data, 0, i_size    * sizeof(float));
    memset(linearGates, 0, lg_size * sizeof(float));

    // ------------------------------------------------------------------
    // Diagonal wavefront scheduler — identical logic to CUDA test()
    //
    // Processes (layer, timestep) tiles in a diagonal wavefront order
    // so that data dependencies are always satisfied:
    //   h(layer, t) depends on h(layer,   t-1)  — recurrence
    //                        and h(layer-1, t  )  — previous layer
    // ------------------------------------------------------------------
    int lStart = 0, lEnd = 0;
    int rStart = 0, rEnd = 0;
    int recurBatchSize = 2;

    while (true) {

        if (lEnd == 0) {
            // First iteration: start at layer 0, timestep 0
            lStart = 0;
            lEnd   = 1;
            rStart = 0;
        } else {
            // Move "up" one layer and "left" by recurBatchSize timesteps
            lStart++;
            lEnd++;
            rStart -= recurBatchSize;

            // Over the top or off the left → reset to layer 0
            if (lEnd > numLayers || rStart < 0) {
                rStart += (lStart + 1) * recurBatchSize;
                lStart  = 0;
                lEnd    = 1;
            }

            // Off the right → step up through layers
            while (rStart >= seqLength && lEnd <= numLayers) {
                lStart++;
                lEnd++;
                rStart -= recurBatchSize;
            }

            // Over the top or off the left → done
            if (lEnd > numLayers || rStart < 0) break;
        }

        rEnd = rStart + recurBatchSize;
        if (rEnd > seqLength) rEnd = seqLength;

        for (int layer = lStart; layer < lEnd; layer++) {
            for (int i = rStart; i < rEnd; i++) {

                elementwise_ref(
                    hiddenSize, miniBatch,
                    tmp_h + 4 * layer * numElements,
                    tmp_i + 4 * i     * numElements,
                    bias  + 8 * layer * hiddenSize,
                    linearGates + 4 * (i * numElements
                                       + layer * seqLength * numElements),
                    h_data + (i + 1) * numElements
                           + layer * (seqLength + 1) * numElements,
                    i_data + i * numElements
                           + (layer + 1) * seqLength * numElements,
                    c_data + i * numElements
                           + layer * (seqLength + 1) * numElements,
                    c_data + (i + 1) * numElements
                           + layer * (seqLength + 1) * numElements);
            }
        }
    }


    // i: top-layer output for all timesteps
    memcpy(testOutputi,
           i_data + numLayers * seqLength * numElements,
           seqLength * numElements * sizeof(float));

    // h, c: final hidden/cell state per layer
    for (int layer = 0; layer < numLayers; layer++) {
        memcpy(testOutputh + layer * numElements,
               h_data + seqLength * numElements
                      + layer * (seqLength + 1) * numElements,
               numElements * sizeof(float));
        memcpy(testOutputc + layer * numElements,
               c_data + seqLength * numElements
                      + layer * (seqLength + 1) * numElements,
               numElements * sizeof(float));
    }

    free(h_data);
    free(i_data);
    free(c_data);
    free(bias);
    free(tmp_h);
    free(tmp_i);
    free(linearGates);
}
