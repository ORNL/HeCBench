#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "main.h"                // (in the main program folder)  needed to recognized input parameters
#include "timer.h"
#include "file.h"

//  UTILITIES
#include "./util/avi/avilib.h"          // (in directory)              needed by avi functions
#include "./util/avi/avimod.h"          // (in directory)              needed by avi functions

//  KERNEL
uint64_t 
kernel_gpu_wrapper(params_common common,
                   int* endoRow,
                   int* endoCol,
                   int* tEndoRowLoc,
                   int* tEndoColLoc,
                   int* epiRow,
                   int* epiCol,
                   int* tEpiRowLoc,
                   int* tEpiColLoc,
                   avi_t* frames);

int 
main(int argc, char* argv []) {

  printf("Workgroup size of kernel = %d \n", NUMBER_THREADS);

  // time
  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;

  // other
  avi_t* frames;

  time0 = get_time();

  //  STRUCTURES, GLOBAL STRUCTURE VARIABLES

  params_common common;
  common.common_mem = sizeof(params_common);

  //   FRAME INFO

  // variables
  char* video_file_name;

  // open movie file
  video_file_name = (char *) "../data/heartwall/test.avi";
  frames = (avi_t*)AVI_open_input_file(video_file_name, 1);                            // added casting
  if (frames == NULL)  {
    AVI_print_error((char *) "Error with AVI_open_input_file");
    return -1;
  }

  // dimensions
  common.no_frames = AVI_video_frames(frames);
  common.frame_rows = AVI_video_height(frames);
  common.frame_cols = AVI_video_width(frames);
  common.frame_elem = common.frame_rows * common.frame_cols;
  common.frame_mem = sizeof(FP) * common.frame_elem;

  time1 = get_time();

  //   CHECK INPUT ARGUMENTS

  if(argc!=2){
    printf("ERROR: missing argument (number of frames to processed) or too many arguments\n");
    return 0;
  }
  else{
    common.frames_processed = atoi(argv[1]);
    if(common.frames_processed<0 || common.frames_processed>common.no_frames){
      printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n", common.frames_processed, common.no_frames);
      return 0;
    }
  }

  time2 = get_time();

  //  INPUTS

  //  READ PARAMETERS FROM FILE

  char* param_file_name = (char *) "../data/heartwall/input.txt";
  read_parameters(param_file_name,  
                  &common.tSize,
                  &common.sSize,
                  &common.maxMove,
                  &common.alpha);

  //  READ SIZE OF INPUTS FROM FILE

  read_header(param_file_name,
              &common.endoPoints,
              &common.epiPoints);

  common.allPoints = common.endoPoints + common.epiPoints;

  //  READ DATA FROM FILE

  //  ENDO POINTS MEMORY ALLOCATION

  common.endo_mem = sizeof(int) * common.endoPoints;

  int* endoRow;
  endoRow = (int*)malloc(common.endo_mem);
  int* endoCol;
  endoCol = (int*)malloc(common.endo_mem);
  int* tEndoRowLoc;
  tEndoRowLoc = (int*)malloc(common.endo_mem * common.no_frames);
  int* tEndoColLoc;
  tEndoColLoc = (int*)malloc(common.endo_mem * common.no_frames);

  //  EPI POINTS MEMORY ALLOCATION

  common.epi_mem = sizeof(int) * common.epiPoints;

  int* epiRow;
  epiRow = (int *)malloc(common.epi_mem);
  int* epiCol;
  epiCol = (int *)malloc(common.epi_mem);
  int* tEpiRowLoc;
  tEpiRowLoc = (int *)malloc(common.epi_mem * common.no_frames);
  int* tEpiColLoc;
  tEpiColLoc = (int *)malloc(common.epi_mem * common.no_frames);

  //  READ DATA FROM FILE

  read_data(param_file_name,
            common.endoPoints,
            endoRow,
            endoCol,
            common.epiPoints,
            epiRow,
            epiCol);

  time3 = get_time();

  //  KERNELL WRAPPER CALL

  uint64_t kernelTime = kernel_gpu_wrapper(common,
                     endoRow,
                     endoCol,
                     tEndoRowLoc,
                     tEndoColLoc,
                     epiRow,
                     epiCol,
                     tEpiRowLoc,
                     tEpiColLoc,
                     frames);

  time4 = get_time();

  //  DUMP DATA TO FILE
#ifdef OUTPUT
  write_data("result.txt",
             common.no_frames,
             common.frames_processed,    
             common.endoPoints,
             tEndoRowLoc,
             tEndoColLoc,
             common.epiPoints,
             tEpiRowLoc,
             tEpiColLoc);

#endif
  //  DEALLOCATION

  // endo points

  free(endoRow);
  free(endoCol);
  free(tEndoRowLoc);
  free(tEndoColLoc);

  // epi points

  free(epiRow);
  free(epiCol);
  free(tEpiRowLoc);
  free(tEpiColLoc);

  time5= get_time();

  //  DISPLAY TIMING

  printf("Time spent in different stages of the application:\n");
  printf("%15.12f s, %15.12f : READ INITIAL VIDEO FRAME\n",
      (FP) (time1-time0) / 1000000, (FP) (time1-time0) / (FP) (time5-time0) * 100);
  printf("%15.12f s, %15.12f : READ COMMAND LINE PARAMETERS\n",
      (FP) (time2-time1) / 1000000, (FP) (time2-time1) / (FP) (time5-time0) * 100);
  printf("%15.12f s, %15.12f : READ INPUTS FROM FILE\n",
      (FP) (time3-time2) / 1000000, (FP) (time3-time2) / (FP) (time5-time0) * 100);
  printf("%15.12f s, %15.12f : GPU ALLOCATION, COPYING, COMPUTATION\n",
      (FP) (time4-time3) / 1000000, (FP) (time4-time3) / (FP) (time5-time0) * 100);
  printf("%15.12f s, %15.12f : GPU KERNELS\n",
      (FP) (kernelTime) / 1000000, (FP) (kernelTime) / (FP) (time5-time0) * 100);
  printf("%15.12f s, %15.12f : FREE MEMORY\n",
      (FP) (time5-time4) / 1000000, (FP) (time5-time4) / (FP) (time5-time0) * 100);
  printf("Total time:\n");
  printf("%15.12f s\n", (FP) (time5-time0) / 1000000);
}
