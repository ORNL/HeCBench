#include "../main.h"                // (in main directory)            needed to recognized input parameters
#include "../util/avi/avilib.h"          // (in directory)              needed by avi functions
#include "../util/avi/avimod.h"          // (in directory)              needed by avi functions
#include "../util/timer/timer.h"
#include <iostream>
#include <omp.h>
#include <math.h>


uint64_t 
kernel_gpu_wrapper(  params_common common,
    int* endoRow,
    int* endoCol,
    int* tEndoRowLoc,
    int* tEndoColLoc,
    int* epiRow,
    int* epiCol,
    int* tEpiRowLoc,
    int* tEpiColLoc,
    avi_t* frames)
{
  // common
  //printf("tSize is %d, sSize is %d\n", common.tSize, common.sSize);
  common.in_rows = common.tSize + 1 + common.tSize;
  common.in_cols = common.in_rows;
  common.in_elem = common.in_rows * common.in_cols;
  common.in_mem = sizeof(fp) * common.in_elem;

  //==================================================50
  // endo points templates
  //==================================================50

  //buffer<fp,1> d_endoT(common.in_elem * common.endoPoints);
  //printf("%d\n", common.in_elem * common.endoPoints);
  fp* endoT = (fp*) malloc (sizeof(fp) * common.in_elem * common.endoPoints);

  //==================================================50
  // epi points templates
  //==================================================50

  //buffer<fp,1> d_epiT(common.in_elem * common.epiPoints);
  fp* epiT = (fp*) malloc (sizeof(fp) * common.in_elem * common.epiPoints);

  //printf("%d\n", common.in_elem * common.epiPoints);

  //====================================================================================================100
  //   AREA AROUND POINT    FROM  FRAME  (LOCAL)
  //====================================================================================================100

  // common
  common.in2_rows = common.sSize + 1 + common.sSize;
  common.in2_cols = common.in2_rows;
  common.in2_elem = common.in2_rows * common.in2_cols;
  common.in2_mem = sizeof(fp) * common.in2_elem;

  // unique
  //buffer<fp,1> d_in2(common.in2_elem * common.allPoints);
  fp* in2 = (fp*) malloc (sizeof(fp) * common.in2_elem * common.allPoints);
  //printf("%d\n", common.in2_elem * common.allPoints);

  //====================================================================================================100
  //   CONVOLUTION  (LOCAL)
  //====================================================================================================100

  // common
  common.conv_rows = common.in_rows + common.in2_rows - 1;                        // number of rows in I
  common.conv_cols = common.in_cols + common.in2_cols - 1;                        // number of columns in I
  common.conv_elem = common.conv_rows * common.conv_cols;                          // number of elements
  common.conv_mem = sizeof(fp) * common.conv_elem;
  common.ioffset = 0;
  common.joffset = 0;

  // unique
  //buffer<fp,1> d_conv(common.conv_elem * common.allPoints);
  //printf("%d\n", common.conv_elem * common.allPoints);
  fp* conv = (fp*) malloc (sizeof(fp) * common.conv_elem * common.allPoints);


  //====================================================================================================100
  //   CUMULATIVE SUM  (LOCAL)
  //====================================================================================================100

  //==================================================50
  //   PADDING OF ARRAY, VERTICAL CUMULATIVE SUM
  //==================================================50

  // common
  common.in2_pad_add_rows = common.in_rows;
  common.in2_pad_add_cols = common.in_cols;

  common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
  common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
  common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
  common.in2_pad_cumv_mem = sizeof(fp) * common.in2_pad_cumv_elem;

  // unique
  //buffer<fp,1> d_in2_pad_cumv(common.in2_pad_cumv_elem * common.allPoints);
  fp* in2_pad_cumv = (fp*) malloc (sizeof(fp) * common.in2_pad_cumv_elem * common.allPoints);
  //printf("%d\n", common.in2_pad_cumv_elem * common.allPoints);

  //==================================================50
  //   SELECTION
  //==================================================50

  // common
  common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;                          // (1 to n+1)
  common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
  common.in2_pad_cumv_sel_collow = 1;
  common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
  common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
  common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
  common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
  common.in2_pad_cumv_sel_mem = sizeof(fp) * common.in2_pad_cumv_sel_elem;

  // unique
  //buffer<fp,1> d_in2_pad_cumv_sel(common.in2_pad_cumv_sel_elem * common.allPoints);
  //printf("%d\n", common.in2_pad_cumv_sel_elem * common.allPoints);
  fp* in2_pad_cumv_sel = (fp*) malloc (sizeof(fp) * common.in2_pad_cumv_sel_elem * common.allPoints);

  //==================================================50
  //   SELECTION  2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
  //==================================================50

  // common
  common.in2_pad_cumv_sel2_rowlow = 1;
  common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
  common.in2_pad_cumv_sel2_collow = 1;
  common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
  common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
  common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
  common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
  common.in2_sub_cumh_mem = sizeof(fp) * common.in2_sub_cumh_elem;

  // unique
  //buffer<fp,1> d_in2_sub_cumh(common.in2_sub_cumh_elem * common.allPoints);
  //printf("%d\n", common.in2_sub_cumh_elem * common.allPoints);
  fp* in2_sub_cumh = (fp*) malloc (sizeof(fp) * common.in2_sub_cumh_elem * common.allPoints);

  //==================================================50
  //   SELECTION
  //==================================================50

  // common
  common.in2_sub_cumh_sel_rowlow = 1;
  common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
  common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
  common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
  common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
  common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
  common.in2_sub_cumh_sel_mem = sizeof(fp) * common.in2_sub_cumh_sel_elem;

  // unique
  //buffer<fp,1> d_in2_sub_cumh_sel(common.in2_sub_cumh_sel_elem * common.allPoints);
  //printf("%d\n", common.in2_sub_cumh_sel_elem * common.allPoints);
  fp* in2_sub_cumh_sel = (fp*) malloc (sizeof(fp) * common.in2_sub_cumh_sel_elem * common.allPoints);

  //==================================================50
  //  SELECTION 2, SUBTRACTION
  //==================================================50

  // common
  common.in2_sub_cumh_sel2_rowlow = 1;
  common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel2_collow = 1;
  common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
  common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
  common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
  common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
  common.in2_sub2_mem = sizeof(fp) * common.in2_sub2_elem;

  // unique
  //buffer<fp,1> d_in2_sub2(common.in2_sub2_elem * common.allPoints);
  //printf("%d\n", common.in2_sub2_elem * common.allPoints);
  fp* in2_sub2 = (fp*) malloc (sizeof(fp) * common.in2_sub2_elem * common.allPoints);

  //====================================================================================================100
  //  CUMULATIVE SUM 2  (LOCAL)
  //====================================================================================================100

  //==================================================50
  //  MULTIPLICATION
  //==================================================50

  // common
  common.in2_sqr_rows = common.in2_rows;
  common.in2_sqr_cols = common.in2_cols;
  common.in2_sqr_elem = common.in2_elem;
  common.in2_sqr_mem = common.in2_mem;

  // unique
  //buffer<fp,1> d_in2_sqr(common.in2_elem * common.allPoints);
  //printf("%d\n", common.in2_elem * common.allPoints);
  fp* in2_sqr = (fp*) malloc (sizeof(fp) * common.in2_elem * common.allPoints);

  //==================================================50
  //  SELECTION 2, SUBTRACTION
  //==================================================50

  // common
  common.in2_sqr_sub2_rows = common.in2_sub2_rows;
  common.in2_sqr_sub2_cols = common.in2_sub2_cols;
  common.in2_sqr_sub2_elem = common.in2_sub2_elem;
  common.in2_sqr_sub2_mem = common.in2_sub2_mem;

  // unique
  //buffer<fp,1> d_in2_sqr_sub2(common.in2_sub2_elem * common.allPoints);
  //printf("%d\n", common.in2_sub2_elem * common.allPoints);
  fp* in2_sqr_sub2 = (fp*) malloc (sizeof(fp) * common.in2_sub2_elem * common.allPoints);

  //====================================================================================================100
  //  FINAL  (LOCAL)
  //====================================================================================================100

  // common
  common.in_sqr_rows = common.in_rows;
  common.in_sqr_cols = common.in_cols;
  common.in_sqr_elem = common.in_elem;
  common.in_sqr_mem = common.in_mem;

  // unique
  //buffer<fp,1> d_in_sqr(common.in_elem * common.allPoints);
  //printf("%d\n", common.in_elem * common.allPoints);
  fp* in_sqr = (fp*) malloc (sizeof(fp) * common.in_elem * common.allPoints);

  //====================================================================================================100
  //  TEMPLATE MASK CREATE  (LOCAL)
  //====================================================================================================100

  // common
  common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
  common.tMask_cols = common.tMask_rows;
  common.tMask_elem = common.tMask_rows * common.tMask_cols;
  common.tMask_mem = sizeof(fp) * common.tMask_elem;

  // unique
  //buffer<fp,1> d_tMask(common.tMask_elem * common.allPoints);
  //printf("%d\n", common.tMask_elem * common.allPoints);
  fp* tMask = (fp*) malloc (sizeof(fp) * common.tMask_elem * common.allPoints);

  //====================================================================================================100
  //  POINT MASK INITIALIZE  (LOCAL)
  //====================================================================================================100

  // common
  common.mask_rows = common.maxMove;
  common.mask_cols = common.mask_rows;
  common.mask_elem = common.mask_rows * common.mask_cols;
  common.mask_mem = sizeof(fp) * common.mask_elem;

  //====================================================================================================100
  //  MASK CONVOLUTION  (LOCAL)
  //====================================================================================================100

  // common
  common.mask_conv_rows = common.tMask_rows;                        // number of rows in I
  common.mask_conv_cols = common.tMask_cols;                        // number of columns in I
  common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;                        // number of elements
  common.mask_conv_mem = sizeof(fp) * common.mask_conv_elem;
  common.mask_conv_ioffset = (common.mask_rows-1)/2;
  if((common.mask_rows-1) % 2 > 0.5){
    common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
  }
  common.mask_conv_joffset = (common.mask_cols-1)/2;
  if((common.mask_cols-1) % 2 > 0.5){
    common.mask_conv_joffset = common.mask_conv_joffset + 1;
  }


  //printf("common.endoPoints=%d\n", common.endoPoints); // 20 
  //printf("common.epiPoints=%d\n", common.epiPoints); // 31
  //printf("common.in_elem=%d\n", common.in_elem);
  //printf("common.endo_mem=%d\n", common.endo_mem); // 80
  //printf("common.epi_mem=%d\n", common.epi_mem); // 124
  //
  //buffer<params_common,1> d_common(&common, 1, props); // range is 1 ?
  //buffer<int,1> d_endoRow(endoRow, common.endoPoints, props);
  //d_endoRow.set_final_data(nullptr);


  //buffer<int,1> d_endoCol(endoCol, common.endoPoints, props);
  //d_endoCol.set_final_data(nullptr);


  //buffer<int,1> d_tEndoRowLoc(tEndoRowLoc, common.endoPoints * common.no_frames, props);

  //buffer<int,1> d_tEndoColLoc(tEndoColLoc, common.endoPoints * common.no_frames, props);
  
  //buffer<int,1> d_epiRow(epiRow, common.epiPoints, props);
  //d_epiRow.set_final_data(nullptr);

  //buffer<int,1> d_epiCol(epiCol, common.epiPoints, props);
  //d_epiCol.set_final_data(nullptr);

  //buffer<int,1> d_tEpiRowLoc(tEpiRowLoc, common.epiPoints * common.no_frames, props);

  //buffer<int,1> d_tEpiColLoc(tEpiColLoc, common.epiPoints * common.no_frames, props);

  fp *mask_conv = (fp*) malloc (sizeof(fp)*common.mask_conv_elem * common.allPoints);
  //buffer<fp,1> d_mask_conv(common.mask_conv_elem * common.allPoints);
  //d_mask_conv.set_final_data(nullptr);

  //printf("%d\n", common.mask_conv_elem * common.allPoints);
  //buffer<fp,1> d_in_mod_temp(common.in_elem * common.allPoints);
  //d_in_mod_temp.set_final_data(nullptr);
  fp *in_mod_temp = (fp*) malloc (sizeof(fp)*common.in_elem * common.allPoints);

  //printf("%d\n", common.in_elem * common.allPoints);
  //buffer<fp,1> d_in_partial_sum(common.in_cols * common.allPoints);
  //d_in_partial_sum.set_final_data(nullptr);
  fp *in_partial_sum = (fp*) malloc (sizeof(fp)*common.in_cols * common.allPoints);

  //printf("%d\n", common.in_cols * common.allPoints);
  //buffer<fp,1> d_in_sqr_partial_sum(common.in_sqr_rows * common.allPoints);
  //d_in_sqr_partial_sum.set_final_data(nullptr);
  fp *in_sqr_partial_sum = (fp*) malloc (sizeof(fp)*common.in_sqr_rows * common.allPoints);

  //printf("%d\n", common.in_sqr_rows * common.allPoints);
  //buffer<fp,1> d_par_max_val(common.mask_conv_rows * common.allPoints);
  //d_par_max_val.set_final_data(nullptr);
  fp *par_max_val = (fp*) malloc (sizeof(fp)*common.mask_conv_rows * common.allPoints);
  
  //printf("%d\n", common.mask_conv_rows * common.allPoints);
  //buffer<int,1> d_par_max_coo( common.mask_conv_rows * common.allPoints);
  //d_par_max_coo.set_final_data(nullptr);
  fp *par_max_coo = (fp*) malloc (sizeof(fp)*common.mask_conv_rows * common.allPoints);

  //buffer<fp,1> d_in_final_sum(common.allPoints);
  //d_in_final_sum.set_final_data(nullptr);
  fp *in_final_sum = (fp*) malloc (sizeof(fp)* common.allPoints);

  //buffer<fp,1> d_in_sqr_final_sum(common.allPoints);
  //d_in_sqr_final_sum.set_final_data(nullptr);
  fp *in_sqr_final_sum = (fp*) malloc (sizeof(fp)* common.allPoints);

  //buffer<fp,1> d_denomT(common.allPoints);
  //d_denomT.set_final_data(nullptr);
  fp *denomT = (fp*) malloc (sizeof(fp)* common.allPoints);

#ifdef TEST_CHECKSUM

  //printf("%d\n", CHECK);
  fp* checksum = (fp*) malloc (sizeof(fp)*CHECK);
#endif

  //====================================================================================================100
  //  EXECUTION PARAMETERS
  //====================================================================================================100

  // All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
  //size_t local_work_size = NUMBER_THREADS;
  //size_t global_work_size = common.allPoints * local_work_size;

#ifdef DEBUG
  printf("# of workgroups = %d, # of threads/workgroup = %d (ensure that device can handle)\n",(int)(global_work_size/local_work_size), (int)local_work_size);
#endif


  printf("frame progress: ");
  fflush(NULL);

  //====================================================================================================100
  //  LAUNCH
  //====================================================================================================100

  // variables
  fp* frame;
  int frame_no;

  //buffer<fp,1> d_frame(common.frame_elem);
  int allPoints = common.allPoints; 

  uint64_t kernel_time = 0;
  
#pragma omp target data map(alloc: endoT[0:common.in_elem * common.endoPoints],\
                                    epiT[0:common.in_elem * common.epiPoints],\
                                    in2[0:common.in2_elem * common.allPoints],\
                                    conv[0:common.conv_elem * common.allPoints],\
                                    in2_pad_cumv[0:common.in2_pad_cumv_elem * common.allPoints],\
                                    in2_pad_cumv_sel[0:common.in2_pad_cumv_sel_elem * common.allPoints],\
                                    in2_sub_cumh[0:common.in2_sub_cumh_elem * common.allPoints],\
                                    in2_sub_cumh_sel[0:common.in2_sub_cumh_sel_elem * common.allPoints],\
                                    in2_sub2[0:common.in2_sub2_elem * common.allPoints],\
                                    in2_sqr[0:common.in2_elem * common.allPoints],\
                                    in2_sqr_sub2[0:common.in2_sqr_sub2_elem * common.allPoints],\
                                    in_sqr[0:common.in_elem * common.allPoints],\
                                    tMask[0:common.tMask_elem * common.allPoints],\
                                    mask_conv[0:common.mask_conv_elem * common.allPoints],\
                                    in_mod_temp[0:common.in_elem * common.allPoints],\
                                    in_partial_sum[0:common.in_cols * common.allPoints],\
                                    in_sqr_partial_sum[0:common.in_sqr_rows * common.allPoints],\
                                    par_max_val[0:common.mask_conv_rows * common.allPoints],\
                                    par_max_coo[0:common.mask_conv_rows * common.allPoints],\
                                    in_final_sum[0:common.allPoints],\
                                    in_sqr_final_sum[0:common.allPoints],\
                                    denomT[0:common.allPoints]) \
                        map(to:     endoRow[0:common.endoPoints],\
                                    endoCol[0:common.endoPoints],\
                                    epiRow[0:common.epiPoints],\
                                    epiCol[0:common.epiPoints])\
                        map(from:   tEndoRowLoc[0:common.endoPoints * common.no_frames],\
                                    tEndoColLoc[0:common.endoPoints * common.no_frames],\
                                    tEpiRowLoc[0:common.epiPoints * common.no_frames],\
                                    tEpiColLoc[0:common.epiPoints * common.no_frames]) 
  {
#ifdef TEST_CHECKSUM
#pragma omp target data map(alloc: checksum[0:CHECK])
#endif

  for(frame_no=0; frame_no<common.frames_processed; frame_no++){

    //==================================================50
    //  get and write current frame to GPU buffer
    //==================================================50

    // Extract a cropped version of the first frame from the video file
    frame = get_frame(  frames,                // pointer to video file
        frame_no,              // number of frame that needs to be returned
        0,                  // cropped?
        0,                  // scaled?
        1);                  // converted

    // copy frame to GPU memory
    #pragma omp target data map(frame[0:common.frame_elem])
    {
      //==================================================50
      //  launch kernel
      //==================================================50
      uint64_t start_time = get_time();
      #pragma omp target teams num_teams(allPoints) thread_limit(NUMBER_THREADS)
      {
        #pragma omp parallel
        {
          #include "kernel.h"
        }
      }
      uint64_t end_time = get_time();
      kernel_time += end_time - start_time;
    }

    // free frame after each loop iteration, since AVI library allocates memory for every frame fetched
    free(frame);

    //==================================================50
    //  print frame progress
    //==================================================50

    // print frame progress
    printf("%d ", frame_no);
    fflush(NULL);

    //==================================================50
    //  DISPLAY CHECKSUM (TESTING)
    //==================================================50

#ifdef TEST_CHECKSUM
    #pragma omp target update from(checksum[0:CHECK])
    printf("CHECKSUM:\n");
    for(int i=0; i<CHECK; i++){
      printf("i=%d checksum=%f\n", i, checksum[i]);
    }
    printf("\n\n");
#endif

    //==================================================50
    //  End
    //==================================================50
  }
  }

#ifdef TEST_CHECKSUM
  free(checksum);
#endif
  free(endoT);
  free(epiT);
  free(in2);
  free(conv);
  free(in2_pad_cumv);
  free(in2_pad_cumv_sel);
  free(in2_sub_cumh);
  free(in2_sub_cumh_sel);
  free(in2_sub2);
  free(in2_sqr);
  free(in2_sqr_sub2);
  free(in_sqr);
  free(tMask);
  free(mask_conv);
  free(in_mod_temp);
  free(in_partial_sum);
  free(in_sqr_partial_sum);
  free(par_max_val);
  free(par_max_coo);
  free(in_final_sum);
  free(in_sqr_final_sum);
  free(denomT);

  //  PRINT FRAME PROGRESS END

  printf("\n");
  fflush(NULL);
  return kernel_time;
}
