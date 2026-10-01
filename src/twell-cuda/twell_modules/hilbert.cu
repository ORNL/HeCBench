#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cassert>

// returns the next power of two greater than or equal to value.
// input: positive integer value. output: rounded power-of-two size.
inline int next_power_of_two(int value)
{
    if (value <= 1) return 1;
    // built-in clz returns the number of 0s before the first 1
    // 32 total bits in an int -> therefore ceil(log2(value))
    int rounded_exponent = 32 - __builtin_clz((uint32_t)(value - 1));
    return 1 << rounded_exponent;
}

// converts a Hilbert linear index into 2D grid coordinates.
// inputs: power-of-two grid side length and linear Hilbert index. output: grid_coord_m/n updated.
inline void update_hilbert_coordinate(
    // power-of-two grid side length
    int hilbert_square_side_len,
    // [0, ..., hilbert_square_side_len^2 - 1]
    int hilbert_linear_idx,
    // 2D coordinates on the hilbert grid
    int& grid_coord_m,
    int& grid_coord_n
)
{
    int running_coord_m = 0;
    int running_coord_n = 0;
    // every iteration, we process our running-coordinate-based index of one of
    // the four quadrants -> from innermost (rightmost bits) to
    // outermost (leftmost bits). First we rotate the running coordinates from
    // the previous iteration (i.e., local within the quadrant) according to the
    // Hilbert curve rules. Then, we add the coordinates of the start of the
    // current quadrant itself. Finally, we consume the rightmost bits, and look
    // at the next two bits for the next iteration.
    int remaining_hilbert_idx = hilbert_linear_idx;
    for(int running_square_side_len = 1;
        running_square_side_len < hilbert_square_side_len;
        running_square_side_len <<= 1)
        {
        // Hilbert row-major index within the current square defining order
        // [0, 1]
        // [3, 2]
        // idxs 0 and 1 are assigned the first row (bit 0)
        // idxs 2 and 3 are assigned the second row (bit 1)
        int quadrant_bit_m = (remaining_hilbert_idx >> 1) & 1;
        // idxs 0 and 3 are assigned the first column (bit 0)
        // idxs 1 and 2 are assigned the second column (bit 1)
        int quadrant_bit_n = (remaining_hilbert_idx ^ quadrant_bit_m) & 1;
        if(quadrant_bit_n == 0) {
            // for idx 1 at coord [0, 0] (bottom left) we reflect the previous
            // running coordinates across the diagonal, exchanging locations of
            // the inner quadrants at idxs 1 and 3 -> where the bits don't match
            // i.e. defining order:
            // [0, 3]
            // [1, 2]
            int tmp = running_coord_m;
            running_coord_m = running_coord_n;
            running_coord_n = tmp;
            // for idx 3 at coord [1, 0] (bottom left) we also rotate the
            // previous running coordinates by a full 180 degrees:
            // i.e., defining an implicit order of the previous inner quadrants:
            // [2, 1]
            // [3, 0]
            if(quadrant_bit_m == 1) {
                running_coord_m = running_square_side_len - 1 - running_coord_m;
                running_coord_n = running_square_side_len - 1 - running_coord_n;
            }
        }
        running_coord_m += quadrant_bit_m * running_square_side_len;
        running_coord_n += quadrant_bit_n * running_square_side_len;
        remaining_hilbert_idx >>= 2;
    }
    grid_coord_m = running_coord_m;
    grid_coord_n = running_coord_n;
}

inline void build_partitioned_hilbert_schedule_for_grid(
    int num_tiles_m,
    int num_tiles_n,
    int num_partitions,
    int schedule_size_per_partition,
    int* output_partitioned_schedule
)
{
    // builds a round-robin partitioned Hilbert traversal schedule for a 2D tile grid.
    // inputs: grid shape, partition count, partition size, output buffer. output: packed tile schedule written.
    assert (num_tiles_m > 0);
    assert (num_tiles_n > 0);
    assert (num_partitions > 0);
    assert (schedule_size_per_partition > 0);
    assert (num_tiles_m <= 65535);
    assert (num_tiles_n <= 65535);

    // fill with -1 to mark empty slots in the schedule such that our kernel
    // can stop iterating through the schedule when it hits -1
    std::fill(
        output_partitioned_schedule,
        output_partitioned_schedule +
            num_partitions * schedule_size_per_partition,
        -1);
    // maximum rectangle side length of the hilbert curve that can fit all tiles
    const int hilbert_square_side_len =
        next_power_of_two(std::max(num_tiles_m, num_tiles_n));

    std::vector<int> current_written_tiles_per_partition(num_partitions, 0);
    const int num_tiles_to_assign = num_tiles_m * num_tiles_n;
    int num_assigned_tiles = 0;
    int current_partition = 0;
    for(int hilbert_linear_idx = 0;
        num_assigned_tiles < num_tiles_to_assign &&
        hilbert_linear_idx < hilbert_square_side_len * hilbert_square_side_len;
        ++hilbert_linear_idx)
        {
        int tile_m, tile_n;
        update_hilbert_coordinate(
            hilbert_square_side_len, hilbert_linear_idx, tile_m, tile_n);
        if (tile_m >= num_tiles_m || tile_n >= num_tiles_n) continue;
        
        int packed_tile_coord = (tile_m << 16) | tile_n;
        
        int current_partition_write_offset =
            current_written_tiles_per_partition[current_partition];
        assert(current_partition_write_offset < schedule_size_per_partition);
        output_partitioned_schedule[
            current_partition * schedule_size_per_partition +
            current_partition_write_offset] = packed_tile_coord;

        current_written_tiles_per_partition[current_partition] = 
            current_partition_write_offset + 1;
        current_partition++;
        if (current_partition == num_partitions) current_partition = 0;
        num_assigned_tiles++;
    }
    assert(num_assigned_tiles == num_tiles_to_assign);
}
