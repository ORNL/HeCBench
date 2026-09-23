#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include "zipf.h"

// Work-group size: 128 work-items (threads) cooperate on one tile of input.
#define WG_SIZE 128
// Each work-item is responsible for up to 32 consecutive input elements.
#define ITEMS_PER_WI 32
// One tile = one work-group's worth of input: 128 * 32 = 4096 elements.
#define TILE (WG_SIZE * ITEMS_PER_WI)

enum class ReduceKind { Sum, Min, Max };

// Compile-time switch for the three reductions. Kind is a template argument,
// so the unused branches are compiled away (if constexpr).
template <ReduceKind Kind>
struct Reducer {
  // Neutral element: combining identity() with x must leave x unchanged.
  //   sum: 0
  //   min: +infinity  (so min(id, x) == x)
  //   max: -infinity  (so max(id, x) == x)
  static constexpr value_t identity()
  {
    if constexpr (Kind == ReduceKind::Sum)
      return 0;
    else if constexpr (Kind == ReduceKind::Min)
      return std::numeric_limits<value_t>::max();
    else
      return std::numeric_limits<value_t>::lowest();
  }

  // Combine two values with the chosen operator.
  static value_t apply(value_t a, value_t b)
  {
    if constexpr (Kind == ReduceKind::Sum)
      return a + b;
    else if constexpr (Kind == ReduceKind::Min)
      return sycl::min(a, b);
    else
      return sycl::max(a, b);
  }

  // Collective reduce: every work-item in group g contributes its v, and
  // every work-item receives the same combined result. Group can be a
  // work-group or a sub-group.
  template <typename Group>
  static value_t over_group(Group g, value_t v)
  {
    if constexpr (Kind == ReduceKind::Sum)
      return sycl::reduce_over_group(g, v, sycl::plus<value_t>());
    else if constexpr (Kind == ReduceKind::Min)
      return sycl::reduce_over_group(g, v, sycl::minimum<value_t>());
    else
      return sycl::reduce_over_group(g, v, sycl::maximum<value_t>());
  }

  // Merge a partial result into a global output slot.
  // A segment that crosses a tile boundary is reduced by several work-groups,
  // so those partials must be combined atomically.
  // For min/max a partial that cannot move the stored value is dropped without
  // an atomic: whatever is stored is itself a partial of the same segment, so
  // the result is unchanged. This keeps the largest groups, which are split
  // over thousands of tiles, off a single contended address.
  static void combine(value_t *addr, value_t v)
  {
    // Relaxed device-wide atomic on global memory. Segmented reduce does not
    // need acquire/release: we wait on the queue after the kernel.
    sycl::atomic_ref<value_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> a(*addr);
    if constexpr (Kind == ReduceKind::Sum) {
      a.fetch_add(v);                 // d_out[s] += v
    } else if constexpr (Kind == ReduceKind::Min) {
      if (a.load() > v)               // skip the atomic if v cannot win
        a.fetch_min(v);               // d_out[s] = min(d_out[s], v)
    } else {
      if (a.load() < v)
        a.fetch_max(v);               // d_out[s] = max(d_out[s], v)
    }
  }
};

// Binary search: first index s in [0, n) with offsets[s] >= pos.
// offsets is a CSR-style array: segment s occupies [offsets[s], offsets[s+1]).
// Calling this with pos = tile start/end locates which segments overlap a tile.
static inline offset_t lower_bound(const offset_t *offsets, offset_t n,
                                   offset_t pos)
{
  offset_t lo = 0, hi = n;
  while (lo < hi) {
    const offset_t mid = lo + ((hi - lo) >> 1); // midpoint, overflow-safe
    if (offsets[mid] < pos)
      lo = mid + 1;                        // pos is to the right of mid
    else
      hi = mid;                            // pos is at or left of mid
  }
  return lo;
}

// One work-group per fixed tile of TILE input elements, so the cost of a
// work-group is independent of the group size distribution. A tile locates its
// segments with a binary search over the offsets: a segment contained in the
// tile is reduced and stored, one crossing a tile boundary is accumulated with
// an atomic. Within a tile, K lanes cooperate on a segment, where K tracks the
// average group size, down to one lane per group for the many tiny groups and
// up to the whole work-group for a group that spans the tile.
template <ReduceKind Kind>
void launch(sycl::queue &q, const value_t *d_in, value_t *d_out,
            const offset_t *d_offsets, offset_t num_segments,
            offset_t num_items)
{
  using R = Reducer<Kind>;

  // -------------------------------------------------------------------------
  // Kernel 1: initialize every output slot to the identity.
  // queue::fill is a memset-style path and is slow for the min/max identities
  // (not 0), so we write them with an explicit kernel.
  // -------------------------------------------------------------------------
  const value_t id = R::identity();
  // Round the global size up to a multiple of WG_SIZE (nd_range requirement).
  const offset_t init_range =
      (num_segments + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
  q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(init_range), sycl::range<1>(WG_SIZE)),
      [=](sycl::nd_item<1> item) {
        const offset_t i = item.get_global_id(0); // one work-item per segment
        if (i < num_segments)                    // extra threads from rounding
          d_out[i] = id;
      });

  const offset_t num_tiles = (num_items + TILE - 1) / TILE; // ceil(N / TILE)

  // -------------------------------------------------------------------------
  // Kernel 2: segmented reduce, one work-group per tile of TILE elements.
  // Cost of a work-group does not depend on how uneven the segment sizes are.
  //
  // A tile finds its segments with a binary search over offsets:
  //   - a segment fully inside the tile is reduced and stored directly
  //   - a segment that crosses a tile boundary is accumulated with an atomic
  //
  // Within a tile, K lanes cooperate on one segment. K tracks the average
  // group size: 1 lane per tiny group, up to the full work-group for a
  // group that spans the whole tile.
  // -------------------------------------------------------------------------
  q.submit([&](sycl::handler &h) {
    // Two segment indices in local (shared) memory: [first, last) of this tile.
    // Only lane 0 computes them; the rest read after a barrier.
    sycl::local_accessor<offset_t, 1> bounds(sycl::range<1>(2), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_tiles * WG_SIZE),
                          sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          const auto g = item.get_group();           // this work-group
          const int tid = item.get_local_id(0);      // 0 .. WG_SIZE-1
          const offset_t t0 = offset_t(g.get_group_id(0)) * TILE; // tile start
          const offset_t t1 = t0 + TILE < num_items ? t0 + TILE : num_items; // exclusive end

          // Lane 0: which segments touch [t0, t1)?
          // bounds[0] = first offset >= t0  (first segment that starts at/after t0,
          //             or the one that started earlier and still overlaps t0)
          // bounds[1] = first offset >= t1
          if (tid == 0) {
            bounds[0] = lower_bound(d_offsets, num_segments + 1, t0);
            bounds[1] = lower_bound(d_offsets, num_segments + 1, t1);
          }
          sycl::group_barrier(g); // wait until bounds[] is visible to all lanes

          // Segments that *start* inside this tile. If a segment started in an
          // earlier tile, its index is seg_lo-1 and we treat it as a "leading
          // partial" below.
          const offset_t seg_lo = bounds[0];
          const offset_t seg_hi = bounds[1] < num_segments ? bounds[1] : num_segments;

          // Leading partial: input [t0, min(offsets[seg_lo], t1)) still belongs
          // to segment (seg_lo-1), which started before this tile.
          if (seg_lo > 0 && d_offsets[seg_lo] > t0) {
            const offset_t end = d_offsets[seg_lo] < t1 ? d_offsets[seg_lo] : t1;
            value_t local = R::identity();
            // Strided walk: lane tid reads t0+tid, t0+tid+WG_SIZE, ...
            for (offset_t i = t0 + tid; i < end; i += WG_SIZE)
              local = R::apply(local, d_in[i]);
            const value_t r = R::over_group(g, local); // combine 128 locals
            if (tid == 0)
              R::combine(&d_out[seg_lo - 1], r); // atomic merge into earlier segment
          }

          const offset_t nseg = seg_hi - seg_lo; // segments that start in this tile
          if (nseg <= 0) return; // tile is only a trailing partial of one earlier segment

          sycl::sub_group sg = item.get_sub_group(); // warp / SIMD group
          const int num_sgs = sg.get_group_linear_range(); // work-group / sg_size

          if (nseg < num_sgs) {
            // Too few segments to give one to each sub-group. Fall back:
            // the whole work-group walks one segment at a time (sequential
            // over segments, parallel over elements inside a segment).
            for (offset_t s = seg_lo; s < seg_hi; s++) {
              const offset_t begin = d_offsets[s];     // first element of s
              const offset_t seg_end = d_offsets[s + 1]; // first element of s+1
              const offset_t end = seg_end < t1 ? seg_end : t1; // clip to tile
              value_t local = R::identity();
              for (offset_t i = begin + tid; i < end; i += WG_SIZE)
                local = R::apply(local, d_in[i]);
              const value_t r = R::over_group(g, local);
              if (tid == 0) {
                if (seg_end <= t1)
                  d_out[s] = r;            // segment finished in this tile
                else
                  R::combine(&d_out[s], r); // continues into the next tile
              }
            }
            return;
          }

          // Enough segments to keep every sub-group busy. Split each
          // sub-group into teams of K lanes; each team reduces one segment.
          const int sg_size = sg.get_local_range()[0]; // typically 16, 32, or 64
          const int avg = static_cast<int>((t1 - t0) / nseg); // mean segment length
          int K = 1;
          // Double K while it is still below avg, stays <= sg_size, and
          // divides sg_size (XOR teams stay complete).
          while (K < avg && (K << 1) <= sg_size && sg_size % (K << 1) == 0)
            K <<= 1;

          const int segs_per_sg = sg_size / K;      // how many teams in one sub-group
          const int lane = sg.get_local_linear_id(); // 0 .. sg_size-1 inside the sub-group
          const int slot = lane / K;      // which team (segment) this lane belongs to
          const int klane = lane & (K - 1); // index of this lane inside its team (0 .. K-1)

          // Grid-stride over segments assigned to this sub-group.
          // Sub-group i starts at segment i * segs_per_sg, then jumps by
          // (number of sub-groups) * (segments per sub-group).
          for (offset_t base = offset_t(sg.get_group_linear_id()) * segs_per_sg;
               base < nseg; base += offset_t(num_sgs) * segs_per_sg) {
            const offset_t idx = base + slot; // this team's segment among the tile's nseg
            const bool active = idx < nseg;  // last round may have idle teams
            // Inactive lanes still need a valid s so they can participate in
            // the shuffles below (shuffle is a collective of the whole sub-group).
            const offset_t s = seg_lo + (active ? idx : 0);
            const offset_t seg_end = d_offsets[s + 1];

            value_t local = R::identity();
            if (active) {
              const offset_t end = seg_end < t1 ? seg_end : t1; // clip to tile
              // Team of K lanes: klane reads begin+klane, begin+klane+K, ...
              for (offset_t i = d_offsets[s] + klane; i < end; i += K)
                local = R::apply(local, d_in[i]);
            }

            // Butterfly reduction inside the team, using sub-group shuffles.
            // permute_group_by_xor(sg, local, off) is the value held by the
            // lane whose id is (my_lane XOR off). All sg_size lanes must
            // execute this, even idle teams.
            // After log2(K) steps, lane klane==0 of each team holds the
            // reduced value of that segment's tile-local piece.
            for (int off = K >> 1; off > 0; off >>= 1)
              local = R::apply(local, sycl::permute_group_by_xor(sg, local, off));

            if (active && klane == 0) {     // only the team leader writes
              if (seg_end <= t1)
                d_out[s] = local;           // whole segment lived in this tile
              else
                R::combine(&d_out[s], local); // trailing partial -> next tile
            }
          }
        });
  });
}

template <ReduceKind Kind>
void bench_op(sycl::queue &q, const char *name, value_t *d_in, value_t *d_out,
              offset_t *d_offsets, offset_t num_segments, offset_t num_items,
              int repeat, const std::vector<value_t> &ref)
{
  launch<Kind>(q, d_in, d_out, d_offsets, num_segments, num_items);
  q.wait();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    launch<Kind>(q, d_in, d_out, d_offsets, num_segments, num_items);
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("%s throughput = %f (G/s)\n", name, 1.f * num_items * repeat / time);

  std::vector<value_t> h_out(num_segments);
  q.memcpy(h_out.data(), d_out, num_segments * sizeof(value_t)).wait();
  count_errors(h_out, ref, name);
}

void segreduce(sycl::queue &q, offset_t num_items, int repeat)
{
  SegProblem p = make_zipf_problem(num_items);
  print_problem(p);

  std::vector<value_t> ref_sum, ref_min, ref_max;
  cpu_segmented(p, ref_sum, ref_min, ref_max);

  value_t *d_in = sycl::malloc_device<value_t>(num_items, q);
  value_t *d_out = sycl::malloc_device<value_t>(p.num_segments, q);
  offset_t *d_offsets = sycl::malloc_device<offset_t>(p.num_segments + 1, q);
  q.memcpy(d_in, p.values.data(), num_items * sizeof(value_t));
  q.memcpy(d_offsets, p.offsets.data(),
           (p.num_segments + 1) * sizeof(offset_t));
  q.wait();

  bench_op<ReduceKind::Sum>(q, "sycl::reduce_over_group Sum", d_in, d_out,
                            d_offsets, p.num_segments, num_items, repeat, ref_sum);
  bench_op<ReduceKind::Min>(q, "sycl::reduce_over_group Min", d_in, d_out,
                            d_offsets, p.num_segments, num_items, repeat, ref_min);
  bench_op<ReduceKind::Max>(q, "sycl::reduce_over_group Max", d_in, d_out,
                            d_offsets, p.num_segments, num_items, repeat, ref_max);

  sycl::free(d_in, q);
  sycl::free(d_out, q);
  sycl::free(d_offsets, q);
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <multiplier> <repeat>\n", argv[0]);
    printf("The total number of elements is 16384 x multiplier\n");
    return 1;
  }
  const int multiplier = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const offset_t num_items = 16384 * offset_t(multiplier);
  if (num_items <= 0)
    return 0;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  segreduce(q, num_items, repeat);
  return 0;
}
