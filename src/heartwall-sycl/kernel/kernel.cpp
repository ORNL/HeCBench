#include "./../main.h"                // (in main directory)            needed to recognized input parameters
#include "./../util/avi/avilib.h"          // (in directory)              needed by avi functions
#include "./../util/avi/avimod.h"          // (in directory)              needed by avi functions
#include "./../util/timer/timer.h"
#include <sycl/sycl.hpp>
#include <iostream>

uint64_t 
kernel_gpu_wrapper(
    params_common common,
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // common
  //printf("tSize is %d, sSize is %d\n", common.tSize, common.sSize);
  common.in_rows = common.tSize + 1 + common.tSize;
  common.in_cols = common.in_rows;
  common.in_elem = common.in_rows * common.in_cols;
  common.in_mem = sizeof(FP) * common.in_elem;

  //==================================================50
  // endo points templates
  //==================================================50

  FP *d_endoT = sycl::malloc_device<FP>(common.in_elem * common.endoPoints, q);
  //printf("%d\n", common.in_elem * common.endoPoints);

  //==================================================50
  // epi points templates
  //==================================================50

  FP *d_epiT = sycl::malloc_device<FP>(common.in_elem * common.epiPoints, q);
  //printf("%d\n", common.in_elem * common.epiPoints);

  //====================================================================================================100
  //   AREA AROUND POINT    FROM  FRAME  (LOCAL)
  //====================================================================================================100

  // common
  common.in2_rows = common.sSize + 1 + common.sSize;
  common.in2_cols = common.in2_rows;
  common.in2_elem = common.in2_rows * common.in2_cols;
  common.in2_mem = sizeof(FP) * common.in2_elem;

  // unique
  FP *d_in2 = sycl::malloc_device<FP>(common.in2_elem * common.allPoints, q);
  //printf("%d\n", common.in2_elem * common.allPoints);

  //====================================================================================================100
  //   CONVOLUTION  (LOCAL)
  //====================================================================================================100

  // common
  common.conv_rows = common.in_rows + common.in2_rows - 1;                        // number of rows in I
  common.conv_cols = common.in_cols + common.in2_cols - 1;                        // number of columns in I
  common.conv_elem = common.conv_rows * common.conv_cols;                          // number of elements
  common.conv_mem = sizeof(FP) * common.conv_elem;
  common.ioffset = 0;
  common.joffset = 0;

  // unique
  FP *d_conv = sycl::malloc_device<FP>(common.conv_elem * common.allPoints, q);
  //printf("%d\n", common.conv_elem * common.allPoints);

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
  common.in2_pad_cumv_mem = sizeof(FP) * common.in2_pad_cumv_elem;

  // unique
  FP *d_in2_pad_cumv = sycl::malloc_device<FP>(common.in2_pad_cumv_elem * common.allPoints, q);
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
  common.in2_pad_cumv_sel_mem = sizeof(FP) * common.in2_pad_cumv_sel_elem;

  // unique
  FP *d_in2_pad_cumv_sel = sycl::malloc_device<FP>(common.in2_pad_cumv_sel_elem * common.allPoints, q);
  //printf("%d\n", common.in2_pad_cumv_sel_elem * common.allPoints);

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
  common.in2_sub_cumh_mem = sizeof(FP) * common.in2_sub_cumh_elem;

  // unique
  FP *d_in2_sub_cumh = sycl::malloc_device<FP>(common.in2_sub_cumh_elem * common.allPoints, q);
  //printf("%d\n", common.in2_sub_cumh_elem * common.allPoints);

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
  common.in2_sub_cumh_sel_mem = sizeof(FP) * common.in2_sub_cumh_sel_elem;

  // unique
  FP *d_in2_sub_cumh_sel = sycl::malloc_device<FP>(common.in2_sub_cumh_sel_elem * common.allPoints, q);
  //printf("%d\n", common.in2_sub_cumh_sel_elem * common.allPoints);

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
  common.in2_sub2_mem = sizeof(FP) * common.in2_sub2_elem;

  // unique
  FP *d_in2_sub2 = sycl::malloc_device<FP>(common.in2_sub2_elem * common.allPoints, q);
  //printf("%d\n", common.in2_sub2_elem * common.allPoints);

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
  FP *d_in2_sqr = sycl::malloc_device<FP>(common.in2_elem * common.allPoints, q);
  //printf("%d\n", common.in2_elem * common.allPoints);

  //==================================================50
  //  SELECTION 2, SUBTRACTION
  //==================================================50

  // common
  common.in2_sqr_sub2_rows = common.in2_sub2_rows;
  common.in2_sqr_sub2_cols = common.in2_sub2_cols;
  common.in2_sqr_sub2_elem = common.in2_sub2_elem;
  common.in2_sqr_sub2_mem = common.in2_sub2_mem;

  // unique
  FP *d_in2_sqr_sub2 = sycl::malloc_device<FP>(common.in2_sub2_elem * common.allPoints, q);
  //printf("%d\n", common.in2_sub2_elem * common.allPoints);

  //====================================================================================================100
  //  FINAL  (LOCAL)
  //====================================================================================================100

  // common
  common.in_sqr_rows = common.in_rows;
  common.in_sqr_cols = common.in_cols;
  common.in_sqr_elem = common.in_elem;
  common.in_sqr_mem = common.in_mem;

  // unique
  FP *d_in_sqr = sycl::malloc_device<FP>(common.in_elem * common.allPoints, q);
  //printf("%d\n", common.in_elem * common.allPoints);

  //====================================================================================================100
  //  TEMPLATE MASK CREATE  (LOCAL)
  //====================================================================================================100

  // common
  common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
  common.tMask_cols = common.tMask_rows;
  common.tMask_elem = common.tMask_rows * common.tMask_cols;
  common.tMask_mem = sizeof(FP) * common.tMask_elem;

  // unique
  FP *d_tMask = sycl::malloc_device<FP>(common.tMask_elem * common.allPoints, q);
  //printf("%d\n", common.tMask_elem * common.allPoints);

  //====================================================================================================100
  //  POINT MASK INITIALIZE  (LOCAL)
  //====================================================================================================100

  // common
  common.mask_rows = common.maxMove;
  common.mask_cols = common.mask_rows;
  common.mask_elem = common.mask_rows * common.mask_cols;
  common.mask_mem = sizeof(FP) * common.mask_elem;

  //====================================================================================================100
  //  MASK CONVOLUTION  (LOCAL)
  //====================================================================================================100

  // common
  common.mask_conv_rows = common.tMask_rows;                        // number of rows in I
  common.mask_conv_cols = common.tMask_cols;                        // number of columns in I
  common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;                        // number of elements
  common.mask_conv_mem = sizeof(FP) * common.mask_conv_elem;
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
  int *d_endoRow = sycl::malloc_device<int>(common.endoPoints, q);
  q.memcpy(d_endoRow, endoRow, common.endo_mem);

  int *d_endoCol = sycl::malloc_device<int>(common.endoPoints, q);
  q.memcpy(d_endoCol, endoCol, common.endo_mem);

  int *d_tEndoRowLoc = sycl::malloc_device<int>(common.endoPoints * common.no_frames, q);
  q.memcpy(d_tEndoRowLoc, tEndoRowLoc, common.endo_mem * common.no_frames);

  int *d_tEndoColLoc = sycl::malloc_device<int>(common.endoPoints * common.no_frames, q);
  q.memcpy(d_tEndoColLoc, tEndoColLoc, common.endo_mem * common.no_frames);

  int *d_epiRow = sycl::malloc_device<int>(common.epiPoints, q);
  q.memcpy(d_epiRow, epiRow, common.epi_mem);

  int *d_epiCol = sycl::malloc_device<int>(common.epiPoints, q);
  q.memcpy(d_epiCol, epiCol, common.epi_mem);

  int *d_tEpiRowLoc = sycl::malloc_device<int>(common.epiPoints * common.no_frames, q);
  q.memcpy(d_tEpiRowLoc, tEpiRowLoc, common.epi_mem * common.no_frames);

  int *d_tEpiColLoc = sycl::malloc_device<int>(common.epiPoints * common.no_frames, q);
  q.memcpy(d_tEpiColLoc, tEpiColLoc, common.epi_mem * common.no_frames);

  FP *d_mask_conv = sycl::malloc_device<FP>(common.mask_conv_elem * common.allPoints, q);
  //printf("%d\n", common.mask_conv_elem * common.allPoints);
  FP *d_in_mod_temp = sycl::malloc_device<FP>(common.in_elem * common.allPoints, q);
  //printf("%d\n", common.in_elem * common.allPoints);
  FP *d_in_partial_sum = sycl::malloc_device<FP>(common.in_cols * common.allPoints, q);
  //printf("%d\n", common.in_cols * common.allPoints);
  FP *d_in_sqr_partial_sum = sycl::malloc_device<FP>(common.in_sqr_rows * common.allPoints, q);
  //printf("%d\n", common.in_sqr_rows * common.allPoints);
  FP *d_par_max_val = sycl::malloc_device<FP>(common.mask_conv_rows * common.allPoints, q);
  //printf("%d\n", common.mask_conv_rows * common.allPoints);
  int *d_par_max_coo = sycl::malloc_device<int>(common.mask_conv_rows * common.allPoints, q);
  FP *d_in_final_sum = sycl::malloc_device<FP>(common.allPoints, q);
  FP *d_in_sqr_final_sum = sycl::malloc_device<FP>(common.allPoints, q);
  FP *d_denomT = sycl::malloc_device<FP>(common.allPoints, q);

#ifdef TEST_CHECKSUM
  FP* checksum = (FP*) malloc (sizeof(FP)*CHECK);
  FP *d_checksum = sycl::malloc_device<FP>(CHECK, q);
  //printf("%d\n", CHECK);
#endif

  //====================================================================================================100
  //  EXECUTION PARAMETERS
  //====================================================================================================100

  // All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
  size_t lws = NUMBER_THREADS;
  size_t gws = common.allPoints * lws;

#ifdef DEBUG
  printf("# of workgroups = %d, # of threads/workgroup = %d (ensure that device can handle)\n",
         (int)(gws/lws), (int)lws);
#endif


  printf("frame progress: ");
  fflush(NULL);

  //====================================================================================================100
  //  LAUNCH
  //====================================================================================================100

  // variables
  FP* frame;
  int frame_no;

  FP *d_frame = sycl::malloc_device<FP>(common.frame_elem, q);

  uint64_t kernel_time = 0;

  for(frame_no=0; frame_no<common.frames_processed; frame_no++){

    //==================================================50
    //  get and write current frame to GPU buffer
    //==================================================50

    // Extract a cropped version of the first frame from the video file
    frame = get_frame(frames, // pointer to video file
        frame_no,             // number of frame that needs to be returned
        0,                  // cropped?
        0,                  // scaled?
        1);                  // converted

    // copy frame to GPU memory
    q.memcpy(d_frame, frame, sizeof(FP) * common.frame_elem).wait();

    //==================================================50
    //  launch kernel
    //==================================================50
    uint64_t start_time = get_time();

    q.submit ([&](sycl::handler &cgh) {
      cgh.parallel_for<class heartwall>(
        sycl::nd_range<1>(sycl::range<1>(gws), sycl::range<1>(lws)),
        [=] (sycl::nd_item<1> item) {
        #include "kernel.sycl"
      });
    });

#ifdef DEBUG
    try {
      q.wait_and_throw();
    } catch (cl::sycl::exception const& e) {
      std::cout << "Caught synchronous SYCL exception:\n"
        << e.what() << std::endl;
    }
#else
    q.wait();
#endif

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
    q.memcpy(checksum, d_checksum, sizeof(FP)*CHECK).wait();
    printf("CHECKSUM:\n");
    for(int i=0; i<CHECK; i++){
      printf("i=%d checksum=%f\n", i, checksum_host[i]);
    }
    printf("\n\n");
#endif

    //==================================================50
    //  End
    //==================================================50

  }

  q.memcpy(tEndoRowLoc, d_tEndoRowLoc, common.endo_mem * common.no_frames);
  q.memcpy(tEndoColLoc, d_tEndoColLoc, common.endo_mem * common.no_frames);
  q.memcpy(tEpiRowLoc, d_tEpiRowLoc, common.epi_mem * common.no_frames);
  q.memcpy(tEpiColLoc, d_tEpiColLoc, common.epi_mem * common.no_frames);
  q.wait();

  //====================================================================================================100
  //  PRINT FRAME PROGRESS END
  //====================================================================================================100
#ifdef TEST_CHECKSUM
  free(checksum);
  sycl::free(d_checksum, q);
#endif
  sycl::free(d_epiT, q);
  sycl::free(d_endoT, q);
  sycl::free(d_in2, q);
  sycl::free(d_conv, q);
  sycl::free(d_in2_pad_cumv, q);
  sycl::free(d_in2_pad_cumv_sel, q);
  sycl::free(d_in2_sub_cumh, q);
  sycl::free(d_in2_sub_cumh_sel, q);
  sycl::free(d_in2_sub2, q);
  sycl::free(d_in2_sqr, q);
  sycl::free(d_in2_sqr_sub2, q);
  sycl::free(d_in_sqr, q);
  sycl::free(d_tMask, q);
  sycl::free(d_endoRow, q);
  sycl::free(d_endoCol, q);
  sycl::free(d_tEndoRowLoc, q);
  sycl::free(d_tEndoColLoc, q);
  sycl::free(d_epiRow, q);
  sycl::free(d_epiCol, q);
  sycl::free(d_tEpiRowLoc, q);
  sycl::free(d_tEpiColLoc, q);
  sycl::free(d_mask_conv, q);
  sycl::free(d_in_mod_temp, q);
  sycl::free(d_in_partial_sum, q);
  sycl::free(d_in_sqr_partial_sum, q);
  sycl::free(d_par_max_val, q);
  sycl::free(d_par_max_coo, q);
  sycl::free(d_in_final_sum, q);
  sycl::free(d_in_sqr_final_sum, q);
  sycl::free(d_denomT, q);
  sycl::free(d_frame, q);

  printf("\n");
  fflush(NULL);
  return kernel_time;
}
