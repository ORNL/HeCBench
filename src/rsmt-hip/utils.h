#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <climits>

static const int MaxPins = 256;  // must be a power of 2

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
static const int WS = 64;  // AMD CDNA wavefront size
using ballot_t = unsigned long long;
__device__ inline int popc_ballot(unsigned long long x) { return __popcll(x); }
#else
static const int WS = 32;  // NVIDIA warp size
using ballot_t = unsigned int;
__device__ inline int popc_ballot(unsigned int x) { return __popc(x); }
#endif

// Number of wavefronts/warps per block for largeNetKernel; keeps total
// thread count at 768 regardless of WS (24*32 on NVIDIA, 12*64 on AMD).
static const int LargeNetWPB = 768 / WS;

using ID = short;  // must be signed (point and edge IDs)
using ctype = int;  // must be signed (coordinates and distances)  // change requires further change below

typedef struct {
  ID src;
  ID dst;
} edge;

static ctype treeLength(const ID num, const ctype* const __restrict__ x, const ctype* const __restrict__ y, const edge* const __restrict__ edges)
{
  // compute wire length of Steiner tree
  ctype len = 0;
  for (ID i = 0; i < num - 1; i++) {
    const ctype x1 = x[edges[i].src];
    const ctype y1 = y[edges[i].src];
    const ctype x2 = x[edges[i].dst];
    const ctype y2 = y[edges[i].dst];
    len += abs(x1 - x2) + abs(y1 - y2);
  }
  return len;
}


// struct to store grid information
struct grid {
  int grid[3];
  int* vc = NULL;
  int* hc = NULL;
  int* min_wid = NULL;
  int* min_space = NULL;
  int* via_space = NULL;
  int llx, lly, tile_wid, tile_height;
};


// struct to store net and pin information
struct net_list
{
  int num_net;
  std::vector<std::tuple<int, int> >* num_net_arr = NULL;
  int* net_id = NULL;
  int* net_num_pins = NULL;
  int* net_min_wid = NULL;
};



// free info from input file
static void free_memory(const grid& g, const net_list& n)
{
  delete [] g.vc;
  delete [] g.hc;
  delete [] g.min_wid;
  delete [] g.min_space;
  delete [] g.via_space;
  delete [] n.net_id;
  delete [] n.net_num_pins;
  delete [] n.net_min_wid;
  delete [] n.num_net_arr;
}

// function to read in input file
static void read_file(const char* file, grid& g, net_list& n)
{
  std::string line;
  std::string text1, text2;
  int line_count = 0;  // for error messages

  std::fstream myfile(file);
  if (!myfile.is_open()) {std::cout << "ERROR: Cannot open input file!\n"; exit(-1);}

  // read grid x and y co-ordinates and number of layers
  getline(myfile, line);
  line_count++;
  std::stringstream data1(line);
  if (!(data1 >> text1 >> g.grid[0] >> g.grid[1] >> g.grid[2])) {std::cout << "ERROR: Line" << line_count << " couldn't be parsed!\n"; exit(-1);}
  if (text1 != "grid") {std::cout << "ERROR: Invalid format of grid!\n"; exit(-1);}
  for (int i = 0; i < 3; i++) {
    if (g.grid[i] < 1) {std::cout << "ERROR: Grid data should be a reasonable number!\n"; exit(-1);}
  }

  // read vertical capacity of each layer
  getline(myfile, line);
  line_count++;
  std::stringstream data2(line);
  g.vc = new int [g.grid[2] + 1];
  if (!(data2 >> text1 >> text2)) {std::cout << "ERROR: Line" << line_count << " couldn't be parsed!\n"; exit(-1);}
  if ((text1 != "vertical") || (text2 != "capacity")) {std::cout << "ERROR: Invalid format of g.vertical capacity!\n"; exit(-1);}
  for (int i = 1; i <= g.grid[2]; i++) {
      if (data2 >> g.vc[i]) {
      if (g.vc[i] < 0) {std::cout << "ERROR: vertical capacity should be a reasonable number!\n"; exit(-1);}
    }
  }

  // read horizontal capacity of each layer
  getline(myfile, line);
  line_count++;
  std::stringstream data3(line);
  g.hc = new int [g.grid[2] + 1];
  if (!(data3 >> text1 >> text2)) {std::cout << "ERROR: Line" << line_count << " couldn't be parsed!\n";}
  if ((text1 != "horizontal") || (text2 != "capacity")) {std::cout << "ERROR: Invalid format of g.horizontal capacity!\n"; exit(-1);}
  for (int i = 1; i <= g.grid[2]; i++) {
    if (data3 >> g.hc[i]) {
      if (g.hc[i] < 0) {std::cout << "ERROR: horizontal capacity should be a reasonable number!\n"; exit(-1);}
    }
  }

  // read minimum width of each layer
  getline(myfile, line);
  line_count++;
  std::stringstream data4(line);
  g.min_wid = new int [g.grid[2] + 1];
  if (!(data4 >> text1 >> text2)) {std::cout << "ERROR: Line" << line_count << " couldn't be parsed!\n"; exit(-1);}
  if ((text1 != "minimum") || (text2 != "width")) {std::cout << "ERROR: Invalid format of minimum width!\n"; exit(-1);}
  for (int i = 1; i <= g.grid[2] + 1; i++) {
    if (data4 >> g.min_wid[i]) {
      if (g.min_wid[i] < 1) {std::cout << "ERROR: Minimum width should be a reasonable number!\n"; exit(-1);}
    }
  }

  // read minimum spacing of each layer
  getline(myfile, line);
  line_count++;
  std::stringstream data5(line);
  g.min_space = new int [g.grid[2] + 1];
  if (!(data5 >> text1 >> text2)) {std::cout << "ERROR: Line" << line_count << " couldn't be parsed!\n"; exit(-1);}
  if ((text1 != "minimum") || (text2 != "spacing")) {std::cout << "ERROR: Invalid format of minimum spacing!\n"; exit(-1);}
  for (int i = 1; i <= g.grid[2]; i++) {
    if (data5 >> g.min_space[i]) {
      if (g.min_space[i] < 0) {std::cout << "ERROR: Minimum spacing should be a reasonable number!\n"; exit(-1);}
    }
  }

  // read via spacing of each layer
  getline(myfile, line);
  line_count++;
  std::stringstream data6(line);
  g.via_space = new int [g.grid[2] + 1];
  if (!(data6 >> text1 >> text2)) {std::cout << "ERROR: Line" << line_count << " couldn't be parsed!\n"; exit(-1);}
  if ((text1 != "via") || (text2 != "spacing")) {std::cout << "ERROR: Invalid format of via spacing!\n"; exit(-1);}
  for (int i = 1; i <= g.grid[2]; i++) {
    if (data6 >> g.via_space[i]) {
      if (g.via_space[i] < 0) {std::cout << "ERROR: Via spacing should be a reasonable number!\n"; exit(-1);}
    }
  }

  // read lower left x and y co-ordinates for the global routing region, tile width and tile height per layer
  getline(myfile, line);
  line_count++;
  std::stringstream data7(line);
  if (!(data7 >> g.llx >> g.lly >> g.tile_wid >> g.tile_height)) {std::cout << "ERROR: Line" << line_count << " couldn't be parsed!\n"; exit(-1);}
  if (g.tile_wid < 1 || g.tile_height < 1) {std::cout << "ERROR: Tile width and tile height should be a reasonable number!\n"; exit(-1);}

  // read total number of nets
  do {getline(myfile, line);} while (line == "");
  line_count++;
  std::stringstream data8(line);
  if (!(data8 >> text1 >> text2 >> n.num_net)) {std::cout << "ERROR: Line" << line_count << " couldn't be parsed!\n"; exit(-1);}
  if ((text1 != "num") || (text2 != "net")) {std::cout << "ERROR: Invalid format of num net!\n"; exit(-1);}
  if (n.num_net < 1) {std::cout << "ERROR: Number of nets should be a reasonable number!\n";}

  // allocate memory
  n.net_id = new int [n.num_net];
  n.net_num_pins = new int [n.num_net];
  n.net_min_wid = new int [n.num_net];
  n.num_net_arr = new std::vector<std::tuple<int, int> > [n.num_net];

  // read net name, net id, number of pins and minimum width of each net
  for (int i = 0; i < n.num_net; i++) {
    getline(myfile, line);
    line_count++;
    std::stringstream data9(line);
    if (!(data9 >> text1 >> n.net_id[i] >> n.net_num_pins[i] >> n.net_min_wid[i])) {std::cout << "ERROR: Line" << line_count << " couldn't be parsed!\n"; exit(-1);}
    if ((n.net_id[i] < 0) || (n.net_num_pins[i] < 2) || (n.net_min_wid[i] < 1)) {std::cout << "ERROR: Net ID, number of pins and min width should be a reasonable number!\n"; exit(-1);}

    // read x, y and layer information of each net
    for (int j = 0; j < n.net_num_pins[i]; j++) {
      getline(myfile, line);
      line_count++;
      std::stringstream data10(line);
      int x, y, layer;
      if (!(data10 >> x >> y >> layer)) {std::cout << "ERROR: Line" << line_count << " couldn't be parsed!\n"; exit(-1);}
      if (!(x >= g.llx && x < (g.grid[0] * g.tile_wid))) x = (g.grid[0] * g.tile_wid) - 1;
      if (!(y >= g.lly && y < (g.grid[1] * g.tile_height))) y = (g.grid[1] * g.tile_height) - 1;
      if (!(layer > 0 && layer <= g.grid[2])) {std::cout << "ERROR: layer should be within grid!\n"; exit(-1);}
      if ((x < 0) || (y < 0) || (x >= INT_MAX / (MaxPins * 4)) || (y >= INT_MAX / (MaxPins * 4))) {std::cout << "ERROR: x or y out of bounds\n"; exit(-1);}
      n.num_net_arr[i].push_back(std::make_tuple(x, y));
    }
  }

  myfile.close();
}


