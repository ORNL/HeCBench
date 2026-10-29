void malvar_he_cutler_demosaic (
  const uint teamX,
  const uint teamY,
  const uint height,
  const uint width,
  const uchar *__restrict__ input_image_p,
  const uint input_image_pitch, 
        uchar *__restrict__ output_image_p,
  const uint output_image_pitch, 
  const int bayer_pattern )
{
  #pragma omp target teams num_teams(teamX * teamY) thread_limit(tile_cols*tile_rows)
  {
  LDSPixelT apron[apron_rows * apron_cols];
  #pragma omp parallel 
  {
  const uint tile_col_blocksize = tile_cols;
  const uint tile_row_blocksize = tile_rows;
  const uint tile_col_block = omp_get_team_num() % teamX;
  const uint tile_row_block = omp_get_team_num() / teamX;
  const int tid = omp_get_thread_num();
  const int nthreads = omp_get_num_threads();

  // The launched team can be smaller than one thread per tile pixel.
  // Stride by the actual team size so every apron sample and pixel is covered.
  for (int apron_fill_task_id = tid; apron_fill_task_id < (int)n_apron_fill_tasks; apron_fill_task_id += nthreads) {
    const uint apron_read_row = (uint)apron_fill_task_id / apron_cols;
    const uint apron_read_col = (uint)apron_fill_task_id % apron_cols;
    const int ag_c = ((int)(apron_read_col + tile_col_block * tile_col_blocksize)) - shalf_ksize;
    const int ag_r = ((int)(apron_read_row + tile_row_block * tile_row_blocksize)) - shalf_ksize;

    apron[apron_read_row * apron_cols + apron_read_col] = tex2D_at(PixelT, input_image, ag_r, ag_c);
  }

  #pragma omp barrier

  for (int tile_flat_id = tid; tile_flat_id < (int)n_tile_pixels; tile_flat_id += nthreads) {
  const uint tile_col = (uint)tile_flat_id % tile_cols;
  const uint tile_row = (uint)tile_flat_id / tile_cols;
  const uint g_c = tile_col_blocksize * tile_col_block + tile_col;
  const uint g_r = tile_row_blocksize * tile_row_block + tile_row;
  const bool valid_pixel_task = (g_r < height) & (g_c < width);

  //valid tasks read from [half_ksize, (tile_rows|tile_cols) + kernel_size - 1)
  const uint a_c = tile_col + half_ksize;
  const uint a_r = tile_row + half_ksize;
  assert_val(a_c >= half_ksize && a_c < apron_cols - half_ksize, a_c);
  assert_val(a_r >= half_ksize && a_r < apron_rows - half_ksize, a_r);

  //note the following formulas are col, row convention and uses i,j - this is done to preserve readability with the originating paper
  const uint i = a_c;
  const uint j = a_r;
#define F(_i, _j) apron_pixel((_j), (_i))

  const int Fij = F(i,j);
  //symmetric 4,2,-1 response - cross
  const int R1 = (4*F(i, j) + 2*(F(i-1,j) + F(i,j-1) + F(i+1,j) + F(i,j+1)) - 
                    F(i-2,j) - F(i+2,j) - F(i,j-2) - F(i,j+2)) / 8;

  //left-right symmetric response - with .5,1,4,5 - theta
  const int R2 = (
      8*(F(i-1,j) + F(i+1,j)) +10*F(i,j) + F(i,j-2) + F(i,j+2)
      - 2*((F(i-1,j-1) + F(i+1,j-1) + F(i-1,j+1) + F(i+1,j+1)) + F(i-2,j) + F(i+2,j))) / 16;

  //top-bottom symmetric response - with .5,1,4,5 - phi
  const int R3 = (
      8*(F(i,j-1) + F(i,j+1)) +10*F(i,j) + F(i-2,j) + F(i+2,j)
      - 2*((F(i-1,j-1) + F(i+1,j-1) + F(i-1,j+1) + F(i+1,j+1)) + F(i,j-2) + F(i,j+2))) / 16;
  //symmetric 3/2s response - checker
  const int R4 = (
      12*F(i,j) - 3*(F(i-2,j) + F(i+2,j) + F(i,j-2) + F(i,j+2))
      + 4*(F(i-1,j-1) + F(i+1,j-1) + F(i-1,j+1) + F(i+1,j+1))) / 16;

  const int G_at_red_or_blue = R1;
  const int R_at_G_in_red = R2;
  const int B_at_G_in_blue = R2;
  const int R_at_G_in_blue = R3;
  const int B_at_G_in_red = R3;
  const int R_at_B = R4;
  const int B_at_R = R4;

#undef F
#undef j
#undef i
  //RGGB -> RedXY = (0, 0), GreenXY1 = (1, 0), GreenXY2 = (0, 1), BlueXY = (1, 1)
  //GRBG -> RedXY = (1, 0), GreenXY1 = (0, 0), GreenXY2 = (1, 1), BlueXY = (0, 1)
  //GBRG -> RedXY = (0, 1), GreenXY1 = (0, 0), GreenXY2 = (1, 1), BlueXY = (1, 0)
  //BGGR -> RedXY = (1, 1), GreenXY1 = (1, 0), GreenXY2 = (0, 1), BlueXY = (0, 0)
  const int r_mod_2 = g_r & 1;
  const int c_mod_2 = g_c & 1;
#define is_rggb (bayer_pattern == RGGB)
#define is_grbg (bayer_pattern == GRBG)
#define is_gbrg (bayer_pattern == GBRG)
#define is_bggr (bayer_pattern == BGGR)

  const int red_col = is_grbg | is_bggr;
  const int red_row = is_gbrg | is_bggr;
  const int blue_col = 1 - red_col;
  const int blue_row = 1 - red_row;

  const int in_red_row = r_mod_2 == red_row;
  const int in_blue_row = r_mod_2 == blue_row;
  const int is_red_pixel = (r_mod_2 == red_row) & (c_mod_2 == red_col);
  const int is_blue_pixel = (r_mod_2 == blue_row) & (c_mod_2 == blue_col);
  const int is_green_pixel = !(is_red_pixel | is_blue_pixel);
  assert(is_green_pixel + is_blue_pixel + is_red_pixel == 1);
  assert(in_red_row + in_blue_row == 1);

  //at R locations: R is original
  //at B locations it is the 3/2s symmetric response
  //at G in red rows it is the left-right symmmetric with 4s
  //at G in blue rows it is the top-bottom symmetric with 4s
  const RGBPixelBaseT R = output_pixel_cast(
      Fij * is_red_pixel +
      R_at_B * is_blue_pixel +
      R_at_G_in_red * (is_green_pixel & in_red_row) +
      R_at_G_in_blue * (is_green_pixel & in_blue_row)
      );
  //at B locations: B is original
  //at R locations it is the 3/2s symmetric response
  //at G in red rows it is the top-bottom symmmetric with 4s
  //at G in blue rows it is the left-right symmetric with 4s
  const RGBPixelBaseT B = output_pixel_cast(
      Fij * is_blue_pixel +
      B_at_R * is_red_pixel +
      B_at_G_in_red * (is_green_pixel & in_red_row) +
      B_at_G_in_blue * (is_green_pixel & in_blue_row)
      );
  //at G locations: G is original
  //at R locations: symmetric 4,2,-1
  //at B locations: symmetric 4,2,-1
  const RGBPixelBaseT G = output_pixel_cast(Fij * is_green_pixel + G_at_red_or_blue * (!is_green_pixel));

  if(valid_pixel_task){
    RGBPixelT output;
#if OUTPUT_CHANNELS == 3 || OUTPUT_CHANNELS == 4
    output.x = R;
    output.y = G;
    output.z = B;
#if OUTPUT_CHANNELS == 4
    output.w = ALPHA_VALUE;
#endif
#else
#error "Unsupported number of output channels"
#endif
    pixel_at(RGBPixelT, output_image, g_r, g_c) = output;
  }
  }
  }
  }
}
