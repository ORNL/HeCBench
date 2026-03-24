/******************************************************************************
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
double run_gmem_atomics(
    sycl::queue &q,
    PixelType *d_image,
    int width,
    int height,
    unsigned int *d_hist,
    bool warmup)
{
    enum
    {
        NUM_PARTS = 1024
    };

    //dim3 block(32, 4);
    //dim3 grid(16, 16);
    //int total_blocks = grid.x * grid.y;
    const int total_blocks = 256;

    // allocate partial histogram
    unsigned int *d_part_hist = sycl::malloc_device<unsigned int>(total_blocks * NUM_PARTS, q);

    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler& cgh) {
      cgh.parallel_for<class hist_gmem_atomics<ACTIVE_CHANNELS, NUM_BINS, PixelType>>(
        sycl::nd_range<2>(sycl::range<2>(64, 512), sycl::range<2>(4, 32)),
        [=] (sycl::nd_item<2> item) {
        int x = item.get_global_id(1);
        int y = item.get_global_id(0);
        int nx = item.get_global_range(1);
        int ny = item.get_global_range(0);
        int t = item.get_local_linear_id();
        int nt = item.get_local_range(1)*item.get_local_range(0);
        int g = item.get_group_linear_id();

        // initialize global memory
        for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS; i += nt)
            d_part_hist[g * NUM_PARTS + i] = 0;

        item.barrier(sycl::access::fence_space::global_and_local);

        // process pixels (updates our group's partial histogram in gmem)
        for (int col = x; col < width; col += nx)
        {
            for (int row = y; row < height; row += ny)
            {
                PixelType pixel = d_image[row * width + col];

                unsigned int bins[ACTIVE_CHANNELS];
                DecodePixel<NUM_BINS>(pixel, bins);

                #pragma unroll
                for (int c = 0; c < ACTIVE_CHANNELS; ++c) {
                   auto ao = sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed, \
                                              sycl::memory_scope::device,\
                                              sycl::access::address_space::global_space>(
                     d_part_hist[g * NUM_PARTS + (NUM_BINS * c) + bins[c]]);
                   ao.fetch_add(1U);
                }
            }
        }

      });
    });

    q.submit([&] (sycl::handler& cgh) {
      cgh.parallel_for<class hist_gmem_accum<ACTIVE_CHANNELS, NUM_BINS, PixelType>>(
        sycl::nd_range<1>(sycl::range<1>((ACTIVE_CHANNELS * NUM_BINS + 127) / 128 * 128), sycl::range<1>(128)),
        [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i > ACTIVE_CHANNELS * NUM_BINS) return; // out of range

        unsigned int total = 0;
        for (int j = 0; j < total_blocks; j++)
            total += d_part_hist[i + NUM_PARTS * j];

        d_hist[i] = total;
      });
    });

    q.wait();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    float elapsed_millis = elapsed_seconds.count()  * 1000;

    sycl::free(d_part_hist, q);

    return elapsed_millis;
}
