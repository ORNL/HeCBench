#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
for (int c = 1; c < NUM/2+1; c++)
{
  int col = (c * 2) - 1;

  int NUM_2 = NUM >> 1;

  // p_i,0 = p_i,1
  pres_black(col, 0) = pres_red(col, 1);
  pres_red(col + 1, 0) = pres_black(col + 1, 1);

  // p_i,jmax+1 = p_i,jmax
  pres_red(col, NUM_2 + 1) = pres_black(col, NUM_2);
  pres_black(col + 1, NUM_2 + 1) = pres_red(col + 1, NUM_2);
}
