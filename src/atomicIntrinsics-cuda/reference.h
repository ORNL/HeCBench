#define min(a,b) (a) < (b) ? (a) : (b)
#define max(a,b) (a) > (b) ? (a) : (b)

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
template <class T>
void computeGold(T *gpuData, const size_t len)
{
  T val = 0;
  bool ok = true;

  for (size_t i = 0; i < len; ++i)
  {
    val += (T)10;
  }

  if (val != gpuData[0])
  {
    printf("Add failed: %d != %d\n", val, gpuData[0]);
    ok = false;
  }

  val = 0;

  for (size_t i = 0; i < len; ++i)
  {
    val -= (T)10;
  }

  if (val != gpuData[1])
  {
    printf("Sub failed: %d != %d\n", val, gpuData[1]);
    ok = false;
  }

  val = (T)(-256);

  for (size_t i = 0; i < len; ++i)
  {
    val = max(val, (T)i);
  }

  if (val != gpuData[2])
  {
    printf("Max failed: %d != %d\n", val, gpuData[2]);
    ok = false;
  }

  val = (T)256;

  for (size_t i = 0; i < len; ++i)
  {
    val = min(val, (T)i);
  }

  if (val != gpuData[3])
  {
    printf("Min failed: %d != %d\n", val, gpuData[3]);
    ok = false;
  }

  val = 0xff;

  for (size_t i = 0; i < len; ++i)
  {
    val &= (T)(2 * i + 7);
  }

  if (val != gpuData[4])
  {
    printf("And failed: %d != %d\n", val, gpuData[4]);
    ok = false;
  }

  val = 0;

  for (size_t i = 0; i < len; ++i)
  {
    val |= (T)(1 << i);
  }

  if (val != gpuData[5])
  {
    printf("Or failed: %d != %d\n", val, gpuData[5]);
    ok = false;
  }

  val = 0xff;

  for (size_t i = 0; i < len; ++i)
  {
    val ^= (T)i;
  }

  if (val != gpuData[6])
  {
    printf("Xor failed: %d != %d\n", val, gpuData[6]);
    ok = false;
  }

  T limit = 17;
  val = 0;

  for (size_t i = 0; i < len; ++i) {
    val = (val >= limit) ? 0 : val + 1;
  }

  if (val != gpuData[7]) {
    printf("atomicInc failed: %d != %d\n", val, gpuData[7]);
    ok = false;
  }

  limit = 137;
  val = 0;

  for (size_t i = 0; i < len; ++i) {
    val = ((val == 0) || (val > limit)) ? limit : val - 1;
  }

  if (val != gpuData[8]) {
    printf("atomicDec failed: %d != %d\n", val, gpuData[8]);
    ok = false;
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
}
