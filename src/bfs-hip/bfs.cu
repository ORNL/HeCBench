#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <hip/hip_runtime.h>

#define MAX_THREADS_PER_BLOCK 256

#include "util.h"

//Structure to hold a node information
struct Node
{
  int starting;
  int no_of_edges;
};

__global__ void
Kernel(const Node* __restrict__ d_graph_nodes, 
       const int* __restrict__ d_graph_edges,
       char* __restrict__ d_graph_mask,
       char* __restrict__ d_updatind_graph_mask,
       const char *__restrict__ d_graph_visited,
       int* __restrict__ d_cost,
       const int no_of_nodes) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if( tid<no_of_nodes && d_graph_mask[tid])
  {
    d_graph_mask[tid]=0;
    const int num_edges = d_graph_nodes[tid].no_of_edges;
    const int starting = d_graph_nodes[tid].starting;

    for(int i=starting; i<(num_edges + starting); i++)
    {
      int id = d_graph_edges[i];
      if(!d_graph_visited[id])
      {
        d_cost[id]=d_cost[tid]+1;
        d_updatind_graph_mask[id]=1;
      }
    }
  }
}

__global__ void
Kernel2(char* __restrict__ d_graph_mask,
        char* __restrict__ d_updatind_graph_mask,
        char* __restrict__ d_graph_visited,
        char* __restrict__ d_over,
        const int no_of_nodes)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if( tid<no_of_nodes && d_updatind_graph_mask[tid])
  {
    d_graph_mask[tid]=1;
    d_graph_visited[tid]=1;
    *d_over=1;
    d_updatind_graph_mask[tid]=0;
  }
}

void run_bfs_cpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size,
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask,
    char *h_graph_visited, int *h_cost_ref)
{
  char stop;
  do{
    //if no thread changes this value then the loop stops
    stop=0;
    for(int tid = 0; tid < no_of_nodes; tid++ )
    {
      if (h_graph_mask[tid] == 1){ 
        h_graph_mask[tid]=0;
        for(int i=h_graph_nodes[tid].starting; 
            i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++){
          int id = h_graph_edges[i];  // node id is connected with node tid
          if(!h_graph_visited[id]){   // if node id has not been visited, enter the body below
            h_cost_ref[id]=h_cost_ref[tid]+1;
            h_updating_graph_mask[id]=1;
          }
        }
      }    
    }

    for(int tid=0; tid< no_of_nodes ; tid++ )
    {
      if (h_updating_graph_mask[tid] == 1){
        h_graph_mask[tid]=1;
        h_graph_visited[tid]=1;
        stop=1;
        h_updating_graph_mask[tid]=0;
      }
    }
  }
  while(stop);
}

void Usage(int argc, char**argv){
  fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);
}

// Returns true if the graph is valid, false otherwise.
bool check_graph(int no_of_nodes, const Node *h_graph_nodes, int edge_list_size,
    const int *h_graph_edges, int source)
{
  if (no_of_nodes <= 0) {
    fprintf(stderr, "Invalid graph: no_of_nodes (%d) must be positive\n", no_of_nodes);
    return false;
  }
  if (edge_list_size <= 0) {
    fprintf(stderr, "Invalid graph: edge_list_size (%d) must be positive\n", edge_list_size);
    return false;
  }
  if (source < 0 || source >= no_of_nodes) {
    fprintf(stderr, "Invalid graph: source (%d) out of range [0, %d)\n", source, no_of_nodes);
    return false;
  }

  // Each node's edge range must lie within the edge list.
  long total_edges = 0;
  for (int i = 0; i < no_of_nodes; i++) {
    int starting = h_graph_nodes[i].starting;
    int num_edges = h_graph_nodes[i].no_of_edges;
    if (num_edges < 0) {
      fprintf(stderr, "Invalid graph: node %d has negative edge count (%d)\n", i, num_edges);
      return false;
    }
    if (starting < 0 || starting > edge_list_size) {
      fprintf(stderr, "Invalid graph: node %d starting offset (%d) out of range [0, %d]\n",
          i, starting, edge_list_size);
      return false;
    }
    if ((long)starting + num_edges > edge_list_size) {
      fprintf(stderr, "Invalid graph: node %d edge range [%d, %ld) exceeds edge_list_size (%d)\n",
          i, starting, (long)starting + num_edges, edge_list_size);
      return false;
    }
    total_edges += num_edges;
  }

  // Every edge must reference a valid node id.
  for (int i = 0; i < edge_list_size; i++) {
    int id = h_graph_edges[i];
    if (id < 0 || id >= no_of_nodes) {
      fprintf(stderr, "Invalid graph: edge %d references node id (%d) out of range [0, %d)\n",
          i, id, no_of_nodes);
      return false;
    }
  }

  printf("Graph check passed (#nodes = %d, #edges in list = %d, edges referenced = %ld)\n",
      no_of_nodes, edge_list_size, total_edges);
  return true;
}

//Apply BFS on a Graph
void run_bfs_gpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size,
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask,
    char *h_graph_visited, int *h_cost)
{

  Node* d_graph_nodes;
  hipMalloc((void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
  hipMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, hipMemcpyHostToDevice) ;

  int* d_graph_edges;
  hipMalloc((void**) &d_graph_edges, sizeof(int)*edge_list_size) ;
  hipMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, hipMemcpyHostToDevice) ;

  char* d_graph_mask;
  hipMalloc((void**) &d_graph_mask, sizeof(char)*no_of_nodes) ;
  hipMemcpy(d_graph_mask, h_graph_mask, sizeof(char)*no_of_nodes, hipMemcpyHostToDevice) ;

  char* d_updating_graph_mask;
  hipMalloc((void**) &d_updating_graph_mask, sizeof(char)*no_of_nodes) ;
  hipMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(char)*no_of_nodes, hipMemcpyHostToDevice) ;

  char* d_graph_visited;
  hipMalloc((void**) &d_graph_visited, sizeof(char)*no_of_nodes) ;
  hipMemcpy(d_graph_visited, h_graph_visited, sizeof(char)*no_of_nodes, hipMemcpyHostToDevice) ;

  int* d_cost;
  hipMalloc((void**) &d_cost, sizeof(int)*no_of_nodes);
  hipMemcpy(d_cost, h_cost, sizeof(int)*no_of_nodes, hipMemcpyHostToDevice) ;

  char h_over;
  char *d_over;
  hipMalloc((void**) &d_over, sizeof(char));

  // setup execution parameters
  dim3 grid((no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);
  dim3 threads(MAX_THREADS_PER_BLOCK);

  long time = 0;
  do {
    h_over = 0;
    hipMemcpy(d_over, &h_over, sizeof(char), hipMemcpyHostToDevice) ;

    auto start = std::chrono::steady_clock::now();

    Kernel<<<grid, threads>>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, 
             d_graph_visited, d_cost, no_of_nodes);
    Kernel2<<<grid, threads>>>(d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    hipMemcpy(&h_over, d_over, sizeof(char), hipMemcpyDeviceToHost) ;
  } while(h_over);

  printf("Total kernel execution time : %f (us)\n", time * 1e-3f);

  // copy result from device to host
  hipMemcpy(h_cost, d_cost, sizeof(int)*no_of_nodes, hipMemcpyDeviceToHost) ;

  hipFree(d_graph_nodes);
  hipFree(d_graph_edges);
  hipFree(d_graph_mask);
  hipFree(d_updating_graph_mask);
  hipFree(d_graph_visited);
  hipFree(d_cost);
  hipFree(d_over);
}

//----------------------------------------------------------
//--cambine:  main function
//--author:    created by Jianbin Fang
//--date:    25/01/2011
//----------------------------------------------------------
int main(int argc, char * argv[])
{
  int no_of_nodes;
  int edge_list_size;
  FILE *fp;
  Node* h_graph_nodes;
  char *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
  char *input_f;
  if(argc!=2){
    Usage(argc, argv);
    exit(0);
  }

  input_f = argv[1];
  printf("Reading File\n");
  //Read in Graph from a file
  fp = fopen(input_f,"r");
  if(!fp){
    printf("Error Reading graph file %s\n", input_f);
    return 1;
  }

  int source = 0;

  fscanf(fp,"%d",&no_of_nodes);

  // allocate host memory
  h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
  h_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
  h_updating_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
  h_graph_visited = (char*) malloc(sizeof(char)*no_of_nodes);

  int start, edgeno;   
  // initalize the memory
  for(int i = 0; i < no_of_nodes; i++){
    fscanf(fp,"%d %d",&start,&edgeno);
    h_graph_nodes[i].starting = start;
    h_graph_nodes[i].no_of_edges = edgeno;
    h_graph_mask[i]=0;
    h_updating_graph_mask[i]=0;
    h_graph_visited[i]=0;
  }
  //read the source node from the file
  fscanf(fp,"%d",&source);
  source=0;
  //set the source node as 1 in the mask
  h_graph_mask[source]=1;
  h_graph_visited[source]=1;
  fscanf(fp,"%d",&edge_list_size);
  int id,cost;
  int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
  for(int i=0; i < edge_list_size ; i++){
    fscanf(fp,"%d",&id);
    fscanf(fp,"%d",&cost);
    h_graph_edges[i] = id;
  }

  if(fp) fclose(fp);    

  // validate the graph before running BFS
  if(!check_graph(no_of_nodes, h_graph_nodes, edge_list_size, h_graph_edges, source)){
    fprintf(stderr, "Graph check failed. Aborting.\n");
    free(h_graph_nodes);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_graph_edges);
    return 1;
  }

  // allocate mem for the result on host side
  int *h_cost = (int*) malloc(sizeof(int)*no_of_nodes);
  int *h_cost_ref = (int*)malloc(sizeof(int)*no_of_nodes);
  for(int i=0;i<no_of_nodes;i++){
    h_cost[i]=-1;
    h_cost_ref[i] = -1;
  }
  h_cost[source]=0;
  h_cost_ref[source]=0;    

  printf("run bfs (#nodes = %d) on device\n", no_of_nodes);
  run_bfs_gpu(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, 
      h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost);  

  printf("run bfs (#nodes = %d) on host (cpu) \n", no_of_nodes);
  // initalize the memory again
  for(int i = 0; i < no_of_nodes; i++){
    h_graph_mask[i]=0;
    h_updating_graph_mask[i]=0;
    h_graph_visited[i]=0;
  }
  //set the source node as 1 in the mask
  source=0;
  h_graph_mask[source]=1;
  h_graph_visited[source]=1;
  run_bfs_cpu(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, 
      h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost_ref);

  // verify
  compare_results<int>(h_cost_ref, h_cost, no_of_nodes);

  free(h_graph_nodes);
  free(h_graph_mask);
  free(h_updating_graph_mask);
  free(h_graph_visited);
  free(h_cost);
  free(h_cost_ref);

  return 0;
}
