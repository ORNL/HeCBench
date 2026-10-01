/*
   Copyright (c) 2016-2020, Texas State University. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/


#ifndef ECL_GRAPH
#define ECL_GRAPH

#include <cstdlib>
#include <cstdio>

struct ECLgraph {
  int nodes;
  int edges;
  int* nindex;
  int* nlist;
  int* eweight;
};

void freeECLgraph(ECLgraph &g)
{
  if (g.nindex != NULL) free(g.nindex);
  if (g.nlist != NULL) free(g.nlist);
  if (g.eweight != NULL) free(g.eweight);
  g.nindex = NULL;
  g.nlist = NULL;
  g.eweight = NULL;
}


ECLgraph readECLgraph(const char* const fname)
{
  ECLgraph g;
  size_t cnt;
  size_t nodes1;  // g.nodes + 1 computed in size_t to avoid overflowing int
  int error_status = 0;
  FILE* f = fopen(fname, "rb");
  if (f == NULL) {
    fprintf(stderr, "ERROR: could not open file %s\n\n", fname);
    exit(-1);
  }

  cnt = fread(&g.nodes, sizeof(g.nodes), 1, f);
  if (cnt != 1 || g.nodes < 1) {
    fprintf(stderr, "ERROR: failed to read valid nodes\n\n");
    fclose(f);
    exit(-1);
  }

  cnt = fread(&g.edges, sizeof(g.edges), 1, f);
  if (cnt != 1 || g.edges < 0) {
    fprintf(stderr, "ERROR: failed to read valid edges\n\n");
    fclose(f);
    exit(-1);
  }

  nodes1 = (size_t)g.nodes + 1;
  g.nindex = (int*)malloc(nodes1 * sizeof(g.nindex[0]));
  g.nlist = (int*)malloc((size_t)g.edges * sizeof(g.nlist[0]));
  g.eweight = (int*)malloc((size_t)g.edges * sizeof(g.eweight[0]));
  if ((g.nindex == NULL) || (g.nlist == NULL) || (g.eweight == NULL)) {
    fprintf(stderr, "ERROR: memory allocation failed\n\n");
    error_status = 1;
    goto release;
  }

  // check g.nindex
  cnt = fread(g.nindex, sizeof(g.nindex[0]), nodes1, f);
  if (cnt != nodes1) {
    fprintf(stderr, "ERROR: failed to read neighbor index list\n\n");
    error_status = 1;
    goto release;
  }
  if (g.nindex[0] != 0) {
    fprintf(stderr, "ERROR: neighbor index list always starts at value 0\n");
    error_status = 1;
    goto release;
  }
  if (g.nindex[g.nodes] != g.edges) {
    fprintf(stderr, "ERROR: final value in the neighbor index list equals the total number of edges\n");
    error_status = 1;
    goto release;
  }
  for (int v = 0; v < g.nodes; v++) {
    if (g.nindex[v] > g.nindex[v+1]) {
      fprintf(stderr, "ERROR: neighbor index list must contain non-decreasing values\n");
      error_status = 1;
      goto release;
    }
  }

  // check g.nlist
  cnt = fread(g.nlist, sizeof(g.nlist[0]), (size_t)g.edges, f);
  if (cnt != (size_t)g.edges) {
    fprintf(stderr, "ERROR: failed to read neighbor list\n\n");
    error_status = 1;
    goto release;
  }
  for (int v = 0; v < g.edges; v++) {
    if ((g.nlist[v] < 0) || (g.nlist[v] >= g.nodes)) {
      fprintf(stderr, "ERROR: value in neighbor list must be a valide node index\n");
      error_status = 1;
      goto release;
    }
  }

  // check g.eweight (cnt = 0 is fine)
  cnt = fread(g.eweight, sizeof(g.eweight[0]), (size_t)g.edges, f);
  if (cnt == 0) {
    free(g.eweight);
    g.eweight = NULL;
  }
  else if (cnt != (size_t)g.edges) {
    error_status = 1;
    fprintf(stderr, "ERROR: failed to read edge weights\n\n");
  }

  release:
  fclose(f);
  if (error_status) {
    freeECLgraph(g);
    exit(-1);
  }

  return g;
}


/* Algorithms such as ECL-MIS require a simple undirected graph: a node that is
   its own neighbor, or an edge that is stored in only one of its two
   directions, makes them spin forever waiting on a node that can never be
   resolved.  Graphs that are legitimately directed (used by, e.g.,
   floydwarshall2) must not be passed to this check. */

void verifyUndirectedECLgraph(ECLgraph &g)
{
  for (int v = 0; v < g.nodes; v++) {
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      if (g.nlist[i] == v) {
        fprintf(stderr, "ERROR: neighbor list must not contain self loops\n");
        freeECLgraph(g);
        exit(-1);
      }
      if ((i > g.nindex[v]) && (g.nlist[i - 1] >= g.nlist[i])) {
        fprintf(stderr, "ERROR: neighbor list of each node must be sorted in increasing order\n");
        freeECLgraph(g);
        exit(-1);
      }
    }
  }

  // the sorted neighbor lists allow the reverse edge to be located by bisection
  for (int v = 0; v < g.nodes; v++) {
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      const int u = g.nlist[i];
      int lo = g.nindex[u];
      int hi = g.nindex[u + 1] - 1;
      int found = 0;
      while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (g.nlist[mid] == v) {
          found = 1;
          break;
        }
        if (g.nlist[mid] < v) lo = mid + 1; else hi = mid - 1;
      }
      if (!found) {
        fprintf(stderr, "ERROR: graph must be undirected, edge %d -> %d is not matched by %d -> %d\n", v, u, u, v);
        freeECLgraph(g);
        exit(-1);
      }
    }
  }
}

#endif
