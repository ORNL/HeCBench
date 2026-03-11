#include <cuda.h>
#include "./../main.h"                // (in main directory)            needed to recognized input parameters
#include "./../util/avi/avilib.h"          // (in directory)              needed by avi functions
#include "./../util/avi/avimod.h"          // (in directory)              needed by avi functions
#include "./../util/timer/timer.h"

// CUDA kernel
#include "kernel.h"

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

  fp* d_endoT;
  cudaMalloc((void**)&d_endoT, common.in_mem * common.endoPoints);
  //printf("%d\n", common.in_elem * common.endoPoints);

  //==================================================50
  // epi points templates
  //==================================================50

  fp* d_epiT;
  cudaMalloc((void**)&d_epiT, common.in_mem * common.epiPoints);

  //====================================================================================================100
  //   AREA AROUND POINT    FROM  FRAME  (LOCAL)
  //====================================================================================================100

  // common
  common.in2_rows = common.sSize + 1 + common.sSize;
  common.in2_cols = common.in2_rows;
  common.in2_elem = common.in2_rows * common.in2_cols;
  common.in2_mem = sizeof(fp) * common.in2_elem;

  fp* d_in2;
  cudaMalloc((void**)&d_in2, common.in2_mem * common.allPoints);
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
  fp* d_conv;
  cudaMalloc((void**)&d_conv, common.conv_mem * common.allPoints);

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
  //printf("%d\n", common.in2_pad_cumv_elem * common.allPoints);
  fp* d_in2_pad_cumv;
  cudaMalloc((void**)&d_in2_pad_cumv, common.in2_pad_cumv_mem * common.allPoints);

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
  fp* d_in2_pad_cumv_sel;
  cudaMalloc((void**)&d_in2_pad_cumv_sel, common.in2_pad_cumv_sel_mem * common.allPoints);

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
  fp* d_in2_sub_cumh;
  cudaMalloc((void**)&d_in2_sub_cumh, common.in2_sub_cumh_mem * common.allPoints);

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
  fp* d_in2_sub_cumh_sel;
  cudaMalloc((void**)&d_in2_sub_cumh_sel, common.in2_sub_cumh_sel_mem * common.allPoints);

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
  fp* d_in2_sub2;
  cudaMalloc((void**)&d_in2_sub2, common.in2_sub2_mem * common.allPoints);

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
  fp* d_in2_sqr;
  cudaMalloc((void**)&d_in2_sqr, common.in2_sqr_mem * common.allPoints);

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
  fp* d_in2_sqr_sub2;
  cudaMalloc((void**)&d_in2_sqr_sub2, common.in2_sqr_sub2_mem * common.allPoints);

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
  fp* d_in_sqr;
  cudaMalloc((void**)&d_in_sqr, common.in_sqr_mem * common.allPoints);

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
  fp* d_tMask;
  cudaMalloc((void**)&d_tMask, common.tMask_mem * common.allPoints);

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


  //printf("common.endPoints=%d\n", common.endoPoints); // 20 
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

  int* d_endoRow;
  cudaMalloc((void**)&d_endoRow, common.endo_mem);
  cudaMemcpy(d_endoRow, endoRow, common.endo_mem, cudaMemcpyHostToDevice);

  int* d_endoCol;
  cudaMalloc((void**)&d_endoCol, common.endo_mem);
  cudaMemcpy(d_endoCol, endoCol, common.endo_mem, cudaMemcpyHostToDevice);

  int* d_tEndoRowLoc;
  int* d_tEndoColLoc;
  cudaMalloc((void**)&d_tEndoRowLoc, common.endo_mem*common.no_frames);
  cudaMemcpy(d_tEndoRowLoc, tEndoRowLoc, common.endo_mem*common.no_frames, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_tEndoColLoc, common.endo_mem*common.no_frames);
  cudaMemcpy(d_tEndoColLoc, tEndoColLoc, common.endo_mem*common.no_frames, cudaMemcpyHostToDevice);

  int* d_epiRow;
  int* d_epiCol;
  cudaMalloc((void**)&d_epiRow, common.epi_mem);
  cudaMemcpy(d_epiRow, epiRow, common.epi_mem, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_epiCol, common.epi_mem);
  cudaMemcpy(d_epiCol, epiCol, common.epi_mem, cudaMemcpyHostToDevice);

  int* d_tEpiRowLoc;
  int* d_tEpiColLoc;
  cudaMalloc((void**)&d_tEpiRowLoc, common.epi_mem*common.no_frames);
  cudaMemcpy(d_tEpiRowLoc, tEpiRowLoc, common.epi_mem*common.no_frames, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_tEpiColLoc, common.epi_mem*common.no_frames);
  cudaMemcpy(d_tEpiColLoc, tEpiColLoc, common.epi_mem*common.no_frames, cudaMemcpyHostToDevice);

  //buffer<fp,1> d_mask_conv(common.mask_conv_elem * common.allPoints);
  //d_mask_conv.set_final_data(nullptr);
  fp* d_mask_conv;
  cudaMalloc((void**)&d_mask_conv, common.mask_conv_mem * common.allPoints);

  //printf("%d\n", common.mask_conv_elem * common.allPoints);
  //buffer<fp,1> d_in_mod_temp(common.in_elem * common.allPoints);
  //d_in_mod_temp.set_final_data(nullptr);
  fp* d_in_mod_temp;
  cudaMalloc((void**)&d_in_mod_temp, common.in_mem * common.allPoints);

  //printf("%d\n", common.in_elem * common.allPoints);
  //buffer<fp,1> d_in_partial_sum(common.in_cols * common.allPoints);
  //d_in_partial_sum.set_final_data(nullptr);

  fp* d_in_partial_sum;
  cudaMalloc((void**)&d_in_partial_sum, sizeof(fp)*common.in_cols * common.allPoints);

  //printf("%d\n", common.in_cols * common.allPoints);
  //buffer<fp,1> d_in_sqr_partial_sum(common.in_sqr_rows * common.allPoints);
  //d_in_sqr_partial_sum.set_final_data(nullptr);

  fp* d_in_sqr_partial_sum;
  cudaMalloc((void**)&d_in_sqr_partial_sum, sizeof(fp)*common.in_sqr_rows * common.allPoints);


  //printf("%d\n", common.in_sqr_rows * common.allPoints);
  //buffer<fp,1> d_par_max_val(common.mask_conv_rows * common.allPoints);
  //d_par_max_val.set_final_data(nullptr);

  fp* d_par_max_val;
  cudaMalloc((void**)&d_par_max_val, sizeof(fp)*common.mask_conv_rows * common.allPoints);

  //printf("%d\n", common.mask_conv_rows * common.allPoints);
  //buffer<int,1> d_par_max_coo( common.mask_conv_rows * common.allPoints);
  //d_par_max_coo.set_final_data(nullptr);

  fp* d_par_max_coo;
  cudaMalloc((void**)&d_par_max_coo, sizeof(fp)*common.mask_conv_rows * common.allPoints);

  //buffer<fp,1> d_in_final_sum(common.allPoints);
  //d_in_final_sum.set_final_data(nullptr);

  fp* d_in_final_sum;
  cudaMalloc((void**)&d_in_final_sum, sizeof(fp)*common.allPoints);

  //buffer<fp,1> d_in_sqr_final_sum(common.allPoints);
  //d_in_sqr_final_sum.set_final_data(nullptr);
  fp* d_in_sqr_final_sum;
  cudaMalloc((void**)&d_in_sqr_final_sum, sizeof(fp)*common.allPoints);

  //buffer<fp,1> d_denomT(common.allPoints);
  //d_denomT.set_final_data(nullptr);

  fp* d_denomT;
  cudaMalloc((void**)&d_denomT, sizeof(fp)*common.allPoints);

#ifdef TEST_CHECKSUM
  //buffer<fp,1> d_checksum(CHECK);
  //d_checksum.set_final_data(nullptr);
  //printf("%d\n", CHECK);
  fp* checksum = (fp*) malloc (sizeof(fp)*CHECK);
  fp* d_checksum;
  cudaMalloc((void**)&d_checksum, sizeof(fp)*CHECK);
#endif

  //====================================================================================================100
  //  EXECUTION PARAMETERS
  //====================================================================================================100

  // All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
  dim3 threads(NUMBER_THREADS);
  dim3 grids(common.allPoints);


  printf("frame progress: ");
  fflush(NULL);

  //====================================================================================================100
  //  LAUNCH
  //====================================================================================================100

  // variables
  fp* frame;
  int frame_no;

  //buffer<fp,1> d_frame(common.frame_elem);
  fp* d_frame;
  cudaMalloc((void**)&d_frame, sizeof(fp)*common.frame_elem);

  uint64_t kernel_time = 0;

  for(frame_no=0; frame_no<common.frames_processed; frame_no++) {

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
    cudaMemcpy(d_frame, frame, sizeof(fp)*common.frame_elem, cudaMemcpyHostToDevice);

    //==================================================50
    //  launch kernel
    //==================================================50
    uint64_t start_time = get_time();

    hw<<<grids, threads>>>(
        frame_no,
        common,
        d_frame, 
        d_endoRow, 
        d_endoCol, 
        d_tEndoRowLoc, 
        d_tEndoColLoc,
        d_epiRow, 
        d_epiCol, 
        d_tEpiRowLoc, 
        d_tEpiColLoc,
        d_endoT,
        d_epiT,
        d_in2,
        d_conv,
        d_in2_pad_cumv,
        d_in2_pad_cumv_sel,
        d_in2_sub_cumh,
        d_in2_sub_cumh_sel,
        d_in2_sub2,
        d_in2_sqr,
        d_in2_sqr_sub2,
        d_in_sqr,
        d_tMask,
        d_mask_conv,
        d_in_mod_temp,
        d_in_partial_sum,
        d_in_sqr_partial_sum,
        d_par_max_val,
        d_par_max_coo,
        d_in_final_sum,
        d_in_sqr_final_sum,
        d_denomT
#ifdef TEST_CHECKSUM
          ,d_checksum
#endif
          );

    cudaDeviceSynchronize();
    uint64_t end_time = get_time();
    kernel_time += end_time - start_time;

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
    cudaMemcpy(checksum, d_checksum, sizeof(fp)*CHECK, cudaMemcpyDeviceToHost);
    printf("CHECKSUM:\n");
    for(int i=0; i<CHECK; i++){
      printf("i=%d checksum=%f\n", i, checksum[i]);
    }
    printf("\n\n");
#endif

  }
  uint64_t end_time = get_time();

  cudaMemcpy(tEndoRowLoc, d_tEndoRowLoc, common.endo_mem * common.no_frames, cudaMemcpyDeviceToHost);
  cudaMemcpy(tEndoColLoc, d_tEndoColLoc, common.endo_mem * common.no_frames, cudaMemcpyDeviceToHost);
  cudaMemcpy(tEpiRowLoc, d_tEpiRowLoc, common.epi_mem * common.no_frames, cudaMemcpyDeviceToHost);
  cudaMemcpy(tEpiColLoc, d_tEpiColLoc, common.epi_mem * common.no_frames, cudaMemcpyDeviceToHost);


  //====================================================================================================100
  //  PRINT FRAME PROGRESS END
  //====================================================================================================100
#ifdef TEST_CHECKSUM
  free(checksum);
  cudaFree(d_checksum);
#endif
  cudaFree(d_epiT);
  cudaFree(d_endoT);
  cudaFree(d_in2);
  cudaFree(d_conv);
  cudaFree(d_in2_pad_cumv);
  cudaFree(d_in2_pad_cumv_sel);
  cudaFree(d_in2_sub_cumh);
  cudaFree(d_in2_sub_cumh_sel);
  cudaFree(d_in2_sub2);
  cudaFree(d_in2_sqr);
  cudaFree(d_in2_sqr_sub2);
  cudaFree(d_in_sqr);
  cudaFree(d_tMask);
  cudaFree(d_endoRow);
  cudaFree(d_endoCol);
  cudaFree(d_tEndoRowLoc);
  cudaFree(d_tEndoColLoc);
  cudaFree(d_epiRow);
  cudaFree(d_epiCol);
  cudaFree(d_tEpiRowLoc);
  cudaFree(d_tEpiColLoc);
  cudaFree(d_mask_conv);
  cudaFree(d_in_mod_temp);
  cudaFree(d_in_partial_sum);
  cudaFree(d_in_sqr_partial_sum);
  cudaFree(d_par_max_val);
  cudaFree(d_par_max_coo);
  cudaFree(d_in_final_sum);
  cudaFree(d_in_sqr_final_sum);
  cudaFree(d_denomT);
  cudaFree(d_frame);

  printf("\n");
  fflush(NULL);
  return kernel_time;
}

