// basic check 
template <typename T>
void verify(T *src, T *dst, int numel) {
  bool ok = true;
  for (int i = 0; i < numel; i++) {
    float s = src[i], d = dst[i]; 
    if (s == 0.25) {
      if (d != 0.f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else if (s == 0.75f || s == 1.25f) {
      if (d != 1.f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else if (s == 1.75f || s == 2.5f) {
      if (d != 2.f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else if (s == 3.5f || s == 5.0f) {
      if (d != 4.f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else if (s > 0.5f && s < 0.75f) {
      if (d != 0.5f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else if (s > 0.f && s < 0.25f) {
      if (d != 0.f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else if (s > 0.25f && s < 0.5f) {
      if (d != 0.5f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else if (s > 0.75f && s < 1.f) {
      if (d != 1.f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else if (s < -0.5f && s > -0.75f) {
      if (d != -0.5f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else if (s < 0.f && s > -0.25f) {
      if (d != -0.f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else if (s < -0.25f && s > -0.5f) {
      if (d != -0.5f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else if (s < -0.75f && s > -1.f) {
      if (d != -1.f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    else {
      if (fabsf(s - d) > 1e-3f) {
        printf("%f %f\n", s, d);
        ok = false;
      }
    }
    if (!ok) break;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
}
