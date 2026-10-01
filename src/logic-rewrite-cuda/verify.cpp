// Framework-neutral C/C++ reference verifier for logic-rewrite (gpuls).
//
// Reusable, as-is, by the CUDA, HIP, and SYCL versions: it depends only on the
// C++ standard library (no CUDA/HIP/SYCL), and operates purely on the AIGER
// (.aig) files that every version reads and writes.
//
// Logic-rewrite transforms (balance/rewrite/refactor/strash/resyn2) are
// equivalence-preserving but heuristic, so the transformed netlist is
// structurally different from the input while remaining functionally
// equivalent. This reference therefore verifies the GPU result two ways:
//   1. structural invariants of the produced AIG (well-formedness), and
//   2. combinational equivalence to the input via bit-parallel random
//      simulation (same random primary-input vectors drive both circuits;
//      primary outputs must match on every vector).
//
// Build (host compiler, no GPU toolchain needed):
//   c++ -O3 -std=c++17 verify.cpp -o verify
// Usage:
//   ./verify <input.aig> <output.aig> [num_vectors]     # default 4096
//   exit code 0 = verified equivalent, non-zero = mismatch / malformed.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>

namespace {

struct Aig {
  int M = 0;              // max variable index
  int I = 0;              // primary inputs
  int O = 0;              // primary outputs
  int A = 0;              // AND gates
  std::vector<int> outs;  // O output literals
  // Per AND gate (indexed by gate order k = 0..A-1, var = I+1+k):
  std::vector<int> rhs0;  // literal of first fanin
  std::vector<int> rhs1;  // literal of second fanin
};

// LEB128-style unsigned decode, matching aigEncodeBinary/aigDecodeBinary.
uint32_t decodeLeb(const unsigned char *buf, size_t &pos) {
  uint32_t x = 0;
  int i = 0;
  unsigned char ch;
  do {
    ch = buf[pos++];
    x |= (uint32_t)(ch & 0x7f) << (7 * i++);
  } while (ch & 0x80);
  return x;
}

bool parseAig(const char *path, Aig &aig) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "verify: cannot open %s\n", path);
    return false;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> data(sz);
  if (fread(data.data(), 1, sz, f) != (size_t)sz) {
    fclose(f);
    fprintf(stderr, "verify: read error on %s\n", path);
    return false;
  }
  fclose(f);

  // Header line: "aig M I L O A"
  int M, I, L, O, Acnt;
  if (sscanf((const char *)data.data(), "aig %d %d %d %d %d", &M, &I, &L, &O,
             &Acnt) != 5) {
    fprintf(stderr, "verify: bad AIGER header in %s\n", path);
    return false;
  }
  if (L != 0) {
    fprintf(stderr, "verify: latches unsupported in %s\n", path);
    return false;
  }
  aig.M = M;
  aig.I = I;
  aig.O = O;
  aig.A = Acnt;

  // Advance past header newline.
  size_t pos = 0;
  while (pos < data.size() && data[pos] != '\n') pos++;
  pos++;

  // O primary-output literals, one ASCII integer per line.
  aig.outs.resize(O);
  for (int i = 0; i < O; i++) {
    aig.outs[i] = atoi((const char *)&data[pos]);
    while (pos < data.size() && data[pos] != '\n') pos++;
    pos++;
  }

  // A AND-gate definitions (binary delta encoding).
  aig.rhs0.resize(Acnt);
  aig.rhs1.resize(Acnt);
  for (int k = 0; k < Acnt; k++) {
    int lhs = 2 * (I + 1 + k);
    uint32_t d0 = decodeLeb(data.data(), pos);
    uint32_t d1 = decodeLeb(data.data(), pos);
    int r0 = lhs - (int)d0;
    int r1 = r0 - (int)d1;
    aig.rhs0[k] = r0;
    aig.rhs1[k] = r1;
  }
  return true;
}

// Well-formedness checks on the produced AIG.
bool checkStructure(const Aig &g, const char *name) {
  bool ok = true;
  if (g.M != g.I + g.A) {
    fprintf(stderr, "verify[%s]: header M(%d) != I(%d)+A(%d)\n", name, g.M, g.I,
            g.A);
    ok = false;
  }
  for (int k = 0; k < g.A && ok; k++) {
    int lhsVar = g.I + 1 + k;
    int v0 = g.rhs0[k] >> 1, v1 = g.rhs1[k] >> 1;
    // Combinational acyclicity: fanins must be earlier variables.
    if (v0 >= lhsVar || v1 >= lhsVar) {
      fprintf(stderr, "verify[%s]: gate %d references non-earlier var\n", name,
              lhsVar);
      ok = false;
    }
    // Canonical fanin ordering produced by the tool: rhs1 <= rhs0.
    if (g.rhs1[k] > g.rhs0[k]) {
      fprintf(stderr, "verify[%s]: gate %d fanins not ordered\n", name, lhsVar);
      ok = false;
    }
  }
  for (int i = 0; i < g.O; i++) {
    if ((g.outs[i] >> 1) > g.M) {
      fprintf(stderr, "verify[%s]: output %d references undefined var\n", name,
              i);
      ok = false;
    }
  }
  return ok;
}

// Bit-parallel simulation over W 64-bit words (=> 64*W random vectors).
// piVals holds I input value-word-blocks; each block is W words.
void simulate(const Aig &g, int W, const std::vector<uint64_t> &piVals,
              std::vector<uint64_t> &poVals) {
  const int nVars = g.M + 1;
  std::vector<uint64_t> val((size_t)nVars * W, 0);  // var 0 = const false = 0
  for (int v = 1; v <= g.I; v++)
    memcpy(&val[(size_t)v * W], &piVals[(size_t)(v - 1) * W],
           W * sizeof(uint64_t));

  for (int k = 0; k < g.A; k++) {
    int gvar = g.I + 1 + k;
    int l0 = g.rhs0[k], l1 = g.rhs1[k];
    const uint64_t *a = &val[(size_t)(l0 >> 1) * W];
    const uint64_t *b = &val[(size_t)(l1 >> 1) * W];
    uint64_t ca = (l0 & 1) ? ~0ULL : 0ULL;
    uint64_t cb = (l1 & 1) ? ~0ULL : 0ULL;
    uint64_t *out = &val[(size_t)gvar * W];
    for (int w = 0; w < W; w++) out[w] = (a[w] ^ ca) & (b[w] ^ cb);
  }

  poVals.assign((size_t)g.O * W, 0);
  for (int i = 0; i < g.O; i++) {
    int lit = g.outs[i];
    const uint64_t *s = &val[(size_t)(lit >> 1) * W];
    uint64_t c = (lit & 1) ? ~0ULL : 0ULL;
    uint64_t *out = &poVals[(size_t)i * W];
    for (int w = 0; w < W; w++) out[w] = s[w] ^ c;
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <input.aig> <output.aig> [num_vectors]\n",
            argv[0]);
    return 2;
  }
  int nVec = (argc > 3) ? atoi(argv[3]) : 4096;
  if (nVec < 64) nVec = 64;
  int W = (nVec + 63) / 64;

  Aig a, b;
  if (!parseAig(argv[1], a) || !parseAig(argv[2], b)) return 2;

  printf("%s: I/O=%d/%d A=%d\n", argv[1], a.I, a.O, a.A);
  printf("%s: I/O=%d/%d A=%d\n", argv[2], b.I, b.O, b.A);

  bool structOk = checkStructure(a, "input") & checkStructure(b, "output");
  if (!structOk) {
    printf("FAIL: structural invariants violated\n");
    return 1;
  }
  if (a.I != b.I || a.O != b.O) {
    printf("FAIL: primary I/O counts differ -> not equivalent\n");
    return 1;
  }

  // Identical random primary-input patterns for both circuits.
  std::mt19937_64 rng(0xC0FFEEULL);
  std::vector<uint64_t> piVals((size_t)a.I * W);
  for (auto &x : piVals) x = rng();

  std::vector<uint64_t> poA, poB;
  simulate(a, W, piVals, poA);
  simulate(b, W, piVals, poB);

  int firstBad = -1;
  for (int i = 0; i < a.O; i++)
    for (int w = 0; w < W; w++)
      if (poA[(size_t)i * W + w] != poB[(size_t)i * W + w]) {
        firstBad = i;
        break;
      }

  printf("simulated %d random input vectors over %d outputs\n", W * 64, a.O);
  if (firstBad >= 0) {
    printf("FAIL: NOT EQUIVALENT (first differing output PO#%d)\n", firstBad);
    return 1;
  }
  printf("PASS: EQUIVALENT (all outputs match on every random vector)\n");
  return 0;
}
