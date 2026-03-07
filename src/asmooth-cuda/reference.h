void reference (int Lx, int Ly, int threshold, int maxRad, 
                float *img, int *box, float *norm, float *out)
{
  float q, sum, s;
  int ksum;

  for (int x = 0; x < Lx; x++) {
    for (int y = 0; y < Ly; y++) {
      sum = 0.f;
      s = q = 1;  // box size
      ksum = 0; // kernel sum

      while (sum < threshold && q < maxRad) {
        s = q;
        sum = 0.f;
        ksum = 0;

        for (int i = -s; i < s+1; i++)
          for (int j = -s; j < s+1; j++)
            if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly) {
              sum += img[(x+i)*Ly+y+j];
              ksum++;
            }
        q++;
      }

      box[x*Ly+y] = s;  // save the box size

      for (int i = -s; i < s+1; i++)
        for (int j = -s; j < s+1; j++)
          if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly)
            if (ksum != 0) norm[(x+i)*Ly+y+j] += 1.f / (float)ksum;
    }
  }

  // normalize the image
  for (int x = 0; x < Lx; x++)
    for (int y = 0; y < Ly; y++) 
      if (norm[x*Ly+y] != 0) img[x*Ly+y] /= norm[x*Ly+y];

  // output file
  for (int x = 0; x < Lx; x++) {
    for (int y = 0; y < Ly; y++) {
      s = box[x*Ly+y];
      sum = 0.f;
      ksum = 0;

      // resmooth with normalized image
      for (int i = -s; i < s+1; i++)
        for (int j = -s; j < s+1; j++) {
          if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly) {
            sum += img[(x+i)*Ly+y+j];
            ksum++;
          }
        }
      if (ksum != 0) out[x*Ly+y] = sum / (float)ksum;
    }
  }
}

void verify (
  const int size,
  const int MaxRad,
  const float* norm,
  const float* h_norm,
  const float* out,
  const float* h_out,
  const   int* box,
  const   int* h_box)
{
  bool ok = true;
  int cnt[10] = {0,0,0,0,0,0,0,0,0,0};
  for (int i = 0; i < size; i++) {
    if (fabsf(norm[i] - h_norm[i]) > 1e-3f) {
      printf("norm: %d %f %f\n", i, norm[i], h_norm[i]);
      ok = false;
      break;
    }
    if (fabsf(out[i] - h_out[i]) > 1e-3f) {
      printf("out: %d %f %f\n", i, out[i], h_out[i]);
      ok = false;
      break;
    }
    if (box[i] != h_box[i]) {
      printf("box: %d %d %d\n", i, box[i], h_box[i]);
      ok = false;
      break;
    } else {
      for (int j = 0; j < MaxRad; j++)
        if (box[i] == j) { cnt[j]++; break; }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
  if (ok) {
    printf("Distribution of box sizes:\n");
    for (int j = 1; j < MaxRad; j++)
      printf("size=%d: %f\n", j, (float)cnt[j]/size);
  }
}


