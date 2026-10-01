//--by Jianbin Fang

#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <sycl/sycl.hpp>
#include "util.h"

#define MAX_THREADS_PER_BLOCK 256

//Structure to hold a node information
struct Node
{
  int starting;
  int no_of_edges;
};

//----------------------------------------------------------
//--bfs on cpu
//--programmer:  jianbin
//--date:  26/01/2011
//--note: width is changed to the new_width
//----------------------------------------------------------
void run_bfs_cpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
    char *h_graph_visited, int *h_cost_ref){
  char stop;
  do{
    //if no thread changes this value then the loop stops
    stop=0;
    for(int tid = 0; tid < no_of_nodes; tid++ )
    {
      if (h_graph_mask[tid] == 1){ 
        h_graph_mask[tid]=0;
        for(int i=h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++){
          int id = h_graph_edges[i];  //--cambine: node id is connected with node tid
          if(!h_graph_visited[id]){  //--cambine: if node id has not been visited, enter the body below
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
//----------------------------------------------------------
//--breadth first search on GPUs
//----------------------------------------------------------
void run_bfs_gpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size,
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask,
    char *h_graph_visited, int *h_cost) noexcept(false) {

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  Node *d_graph_nodes = sycl::malloc_device<Node>(no_of_nodes, q);
  q.memcpy(d_graph_nodes, h_graph_nodes, no_of_nodes * sizeof(Node));

  int *d_graph_edges = sycl::malloc_device<int>(edge_list_size, q); 
  q.memcpy(d_graph_edges, h_graph_edges, edge_list_size * sizeof(int));

  char *d_graph_mask = sycl::malloc_device<char>(no_of_nodes, q);
  q.memcpy(d_graph_mask, h_graph_mask, no_of_nodes * sizeof(char));

  char *d_updating_graph_mask = sycl::malloc_device<char>(no_of_nodes, q);
  q.memcpy(d_updating_graph_mask, h_updating_graph_mask, no_of_nodes * sizeof(char));

  char *d_graph_visited = sycl::malloc_device<char>(no_of_nodes, q);
  q.memcpy(d_graph_visited, h_graph_visited, no_of_nodes * sizeof(char));

  int *d_cost= sycl::malloc_device<int>(no_of_nodes, q);
  q.memcpy(d_cost, h_cost, no_of_nodes * sizeof(int));

  char h_over;
  char *d_over = sycl::malloc_device<char>(1, q);

  int global_work_size = (no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK * MAX_THREADS_PER_BLOCK;
  sycl::range<1> gws (global_work_size);
  sycl::range<1> lws (MAX_THREADS_PER_BLOCK);

  long time = 0;
  do {
    h_over = 0;
    q.memcpy(d_over, &h_over, sizeof(char)).wait();
    
    auto start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class kernel1>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid<no_of_nodes && d_graph_mask[tid]) {
          d_graph_mask[tid]=0;
          const int num_edges = d_graph_nodes[tid].no_of_edges;
          const int starting = d_graph_nodes[tid].starting;

          for(int i=starting; i<(num_edges + starting); i++) {
            int id = d_graph_edges[i];
            if(!d_graph_visited[id]){
              d_cost[id]=d_cost[tid]+1;
              d_updating_graph_mask[id]=1;
            }
          }
        }    
      });
    });

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class kernel2>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid<no_of_nodes && d_updating_graph_mask[tid]) {
          d_graph_mask[tid]=1;
          d_graph_visited[tid]=1;
          d_over[0]=1;
          d_updating_graph_mask[tid]=0;
        }
      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    q.memcpy(&h_over, d_over, sizeof(char)).wait();
  }
  while (h_over);

  printf("Total kernel execution time : %f (us)\n", time * 1e-3f);

  q.memcpy(h_cost, d_cost, sizeof(int)*no_of_nodes).wait();

  free(d_graph_nodes, q);
  free(d_graph_edges, q);
  free(d_graph_mask, q);
  free(d_updating_graph_mask, q);
  free(d_graph_visited, q);
  free(d_cost, q);
  free(d_over, q);
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
  int *h_cost_ref = (int*) malloc(sizeof(int)*no_of_nodes);
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
