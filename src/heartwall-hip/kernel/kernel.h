__global__ void hw (
  const int frame_no,
  const params_common d_common,
  const fp* __restrict__ d_frame,
  int* __restrict__ d_endoRow,
  int* __restrict__ d_endoCol,
  int* __restrict__ d_tEndoRowLoc,
  int* __restrict__ d_tEndoColLoc,
  int* __restrict__ d_epiRow,
  int* __restrict__ d_epiCol,
  int* __restrict__ d_tEpiRowLoc,
  int* __restrict__ d_tEpiColLoc,
  fp* __restrict__ d_endoT,
  fp* __restrict__ d_epiT,
  fp* __restrict__ d_in2,
  fp* __restrict__ d_conv,
  fp* __restrict__ d_in2_pad_cumv,
  fp* __restrict__ d_in2_pad_cumv_sel,
  fp* __restrict__ d_in2_sub_cumh,
  fp* __restrict__ d_in2_sub_cumh_sel,
  fp* __restrict__ d_in2_sub2,
  fp* __restrict__ d_in2_sqr,
  fp* __restrict__ d_in2_sqr_sub2,
  fp* __restrict__ d_in_sqr,
  fp* __restrict__ d_tMask,
  fp* __restrict__ d_mask_conv,
  fp* __restrict__ d_in_mod_temp,
  fp* __restrict__ d_in_partial_sum,
  fp* __restrict__ d_in_sqr_partial_sum,
  fp* __restrict__ d_par_max_val,
  fp* __restrict__ d_par_max_coo,
  fp* __restrict__ d_in_final_sum,
  fp* __restrict__ d_in_sqr_final_sum,
  fp* __restrict__ d_denomT
#ifdef TEST_CHECKSUM
  , fp* __restrict__ checksum
#endif
)
{

	// __global fp* d_in;
	int rot_row;
	int rot_col;
	int in2_rowlow;
	int in2_collow;
	int ic;
	int jc;
	int jp1;
	int ja1, ja2;
	int ip1;
	int ia1, ia2;
	int ja, jb;
	int ia, ib;
	fp s;
	int i;
	int j;
	int row;
	int col;
	int ori_row;
	int ori_col;
	int position;
	fp sum;
	int pos_ori;
	fp temp;
	fp temp2;
	int location;
	int cent;
	int tMask_row; 
	int tMask_col;
	fp largest_value_current = 0;
	fp largest_value = 0;
	int largest_coordinate_current = 0;
	int largest_coordinate = 0;
	fp fin_max_val = 0;
	int fin_max_coo = 0;
	int largest_row;
	int largest_col;
	int offset_row;
	int offset_col;
	fp mean;
	fp mean_sqr;
	fp variance;
	fp deviation;
	int pointer;
	int ori_pointer;
	int loc_pointer;


	//======================================================================================================================================================150
	//	BLOCK/THREAD IDs
	//======================================================================================================================================================150

  int bx = blockIdx.x;  // get current horizontal block index (0-n)
  int tx = threadIdx.x;	// get current horizontal thread index (0-n)
	int ei_new;

	//======================================================================================================================================================150
	//	UNIQUE STRUCTURE RECONSTRUCTED HERE
	//======================================================================================================================================================150

	// common

	// offsets for either endo or epi points (separate arrays for endo and epi points)
	int d_unique_point_no = bx < d_common.endoPoints ? bx : bx-d_common.endoPoints;

	int* d_unique_d_Row = bx < d_common.endoPoints ? d_endoRow: d_epiRow;
	int* d_unique_d_Col = bx < d_common.endoPoints ? d_endoCol: d_epiCol;
	int* d_unique_d_tRowLoc = bx < d_common.endoPoints ? d_tEndoRowLoc: d_tEpiRowLoc;
	int* d_unique_d_tColLoc = bx < d_common.endoPoints ? d_tEndoColLoc: d_tEpiColLoc;
	fp*  d_in = bx < d_common.endoPoints ? &d_endoT[d_unique_point_no * d_common.in_elem] :
                                               &d_epiT[d_unique_point_no * d_common.in_elem] ;
  

	// offsets for all points (one array for all points)
	fp*  d_unique_d_in2 = &d_in2[bx*d_common.in2_elem];
	fp*  d_unique_d_conv = &d_conv[bx*d_common.conv_elem];
	fp*  d_unique_d_in2_pad_cumv = &d_in2_pad_cumv[bx*d_common.in2_pad_cumv_elem];
	fp*  d_unique_d_in2_pad_cumv_sel = &d_in2_pad_cumv_sel[bx*d_common.in2_pad_cumv_sel_elem];
	fp*  d_unique_d_in2_sub_cumh = &d_in2_sub_cumh[bx*d_common.in2_sub_cumh_elem];
	fp*  d_unique_d_in2_sub_cumh_sel = &d_in2_sub_cumh_sel[bx*d_common.in2_sub_cumh_sel_elem];
	fp*  d_unique_d_in2_sub2 = &d_in2_sub2[bx*d_common.in2_sub2_elem];
	fp*  d_unique_d_in2_sqr = &d_in2_sqr[bx*d_common.in2_sqr_elem];
	fp*  d_unique_d_in2_sqr_sub2 = &d_in2_sqr_sub2[bx*d_common.in2_sqr_sub2_elem];
	fp*  d_unique_d_in_sqr = &d_in_sqr[bx*d_common.in_sqr_elem];
	fp*  d_unique_d_tMask = &d_tMask[bx*d_common.tMask_elem];
	fp*  d_unique_d_mask_conv = &d_mask_conv[bx*d_common.mask_conv_elem];

	// used to be local
	fp*  d_unique_d_in_mod_temp = &d_in_mod_temp[bx*d_common.in_elem];
	fp*  d_unique_d_in_partial_sum = &d_in_partial_sum[bx*d_common.in_cols];
	fp*  d_unique_d_in_sqr_partial_sum = &d_in_sqr_partial_sum[bx*d_common.in_sqr_rows];
	fp*  d_unique_d_par_max_val = &d_par_max_val[bx*d_common.mask_conv_rows];
	fp*  d_unique_d_par_max_coo = &d_par_max_coo[bx*d_common.mask_conv_rows];

	fp*  d_unique_d_in_final_sum = &d_in_final_sum[bx];
	fp*  d_unique_d_in_sqr_final_sum = &d_in_sqr_final_sum[bx];
	fp*  d_unique_d_denomT = &d_denomT[bx];

	//======================================================================================================================================================150
	//	END
	//======================================================================================================================================================150

	//======================================================================================================================================================150
	//	Initialize checksum
	//======================================================================================================================================================150
#ifdef TEST_CHECKSUM
	if(bx==0 && tx==0){

		for(i=0; i<CHECK; i++){
			checksum[i] = 0;
		}

	}
#endif
	//======================================================================================================================================================150
	//	INITIAL COORDINATE AND TEMPLATE UPDATE
	//======================================================================================================================================================150

	// generate templates based on the first frame only
	if(frame_no == 0){

		//====================================================================================================100
		//	UPDATE ROW LOC AND COL LOC
		//====================================================================================================100

		// uptade temporary endo/epi row/col coordinates (in each block corresponding to point, narrow work to one thread)
		ei_new = tx;
		if(ei_new == 0){

			// update temporary row/col coordinates
			pointer = d_unique_point_no*d_common.no_frames+frame_no;
			d_unique_d_tRowLoc[pointer] = d_unique_d_Row[d_unique_point_no];
			d_unique_d_tColLoc[pointer] = d_unique_d_Col[d_unique_point_no];

		}

		//====================================================================================================100
		//	CREATE TEMPLATES
		//====================================================================================================100

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = d_unique_d_Row[d_unique_point_no] - 25 + row - 1;
			ori_col = d_unique_d_Col[d_unique_point_no] - 25 + col - 1;
			ori_pointer = ori_col*d_common.frame_rows+ori_row;

			// update template
			d_in[col*d_common.in_rows+row] = d_frame[ori_pointer];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		__syncthreads();

		//====================================================================================================100
		//	checksum
		//====================================================================================================100
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[0] = checksum[0]+d_in[i];
			}
		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		__syncthreads();
#endif
		//====================================================================================================100
		//	End
		//====================================================================================================100

	}

	//======================================================================================================================================================150
	//	PROCESS POINTS
	//======================================================================================================================================================150

	// process points in all frames except for the first one
	if(frame_no != 0){

		//====================================================================================================100
		//	Initialize frame-specific variables
		//====================================================================================================100

		//====================================================================================================100
		//	SELECTION
		//====================================================================================================100

		in2_rowlow = d_unique_d_Row[d_unique_point_no] - d_common.sSize;													// (1 to n+1)
		in2_collow = d_unique_d_Col[d_unique_point_no] - d_common.sSize;

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_rows == 0){
				row = d_common.in2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + in2_rowlow - 1;
			ori_col = col + in2_collow - 1;
			d_unique_d_in2[ei_new] = d_frame[ori_col*d_common.frame_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		__syncthreads();

		//====================================================================================================100
		//	checksum
		//====================================================================================================100
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_elem; i++){
				checksum[1] = checksum[1]+d_unique_d_in2[i];
			}
		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		__syncthreads();
#endif
		//====================================================================================================100
		//	CONVOLUTION
		//====================================================================================================100

		//==================================================50
		//	ROTATION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){
		// while(ei_new < 1){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}

			// execution
			rot_row = (d_common.in_rows-1) - row;
			rot_col = (d_common.in_rows-1) - col;
			d_unique_d_in_mod_temp[ei_new] = d_in[rot_col*d_common.in_rows+rot_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[2] = checksum[2]+d_unique_d_in_mod_temp[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	ACTUAL CONVOLUTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.conv_elem){

			// figure out row/col location in array
			ic = (ei_new+1) % d_common.conv_rows;												// (1-n)
			jc = (ei_new+1) / d_common.conv_rows + 1;											// (1-n)
			if((ei_new+1) % d_common.conv_rows == 0){
				ic = d_common.conv_rows;
				jc = jc-1;
			}

			//
			j = jc + d_common.joffset;
			jp1 = j + 1;
			if(d_common.in2_cols < jp1){
				ja1 = jp1 - d_common.in2_cols;
			}
			else{
				ja1 = 1;
			}
			if(d_common.in_cols < j){
				ja2 = d_common.in_cols;
			}
			else{
				ja2 = j;
			}

			i = ic + d_common.ioffset;
			ip1 = i + 1;
			
			if(d_common.in2_rows < ip1){
				ia1 = ip1 - d_common.in2_rows;
			}
			else{
				ia1 = 1;
			}
			if(d_common.in_rows < i){
				ia2 = d_common.in_rows;
			}
			else{
				ia2 = i;
			}

			s = 0;

			for(ja=ja1; ja<=ja2; ja++){
				jb = jp1 - ja;
				for(ia=ia1; ia<=ia2; ia++){
					ib = ip1 - ia;
					s = s + d_unique_d_in_mod_temp[d_common.in_rows*(ja-1)+ia-1] * d_unique_d_in2[d_common.in2_rows*(jb-1)+ib-1];
				}
			}

			//d_unique_d_conv[d_common.conv_rows*(jc-1)+ic-1] = s;
			d_unique_d_conv[ei_new] = s;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.conv_elem; i++){
				checksum[3] = checksum[3]+d_unique_d_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		// 	CUMULATIVE SUM	(LOCAL)
		//====================================================================================================100

		//==================================================50
		//	PADD ARRAY
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_elem){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_rows == 0){
				row = d_common.in2_pad_cumv_rows - 1;
				col = col-1;
			}

			// execution
			if(	row > (d_common.in2_pad_add_rows-1) &&														// do if has numbers in original array
				row < (d_common.in2_pad_add_rows+d_common.in2_rows) && 
				col > (d_common.in2_pad_add_cols-1) && 
				col < (d_common.in2_pad_add_cols+d_common.in2_cols)){
				ori_row = row - d_common.in2_pad_add_rows;
				ori_col = col - d_common.in2_pad_add_cols;
				d_unique_d_in2_pad_cumv[ei_new] = d_unique_d_in2[ori_col*d_common.in2_rows+ori_row];
			}
			else{																			// do if otherwise
				d_unique_d_in2_pad_cumv[ei_new] = 0;
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[4] = checksum[4]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	VERTICAL CUMULATIVE SUM
		//==================================================50

		//work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_cols){

			// figure out column position
			pos_ori = ei_new*d_common.in2_pad_cumv_rows;

			// variables
			sum = 0;
			
			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_pad_cumv_rows; position = position + 1){
				d_unique_d_in2_pad_cumv[position] = d_unique_d_in2_pad_cumv[position] + sum;
				sum = d_unique_d_in2_pad_cumv[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_cols; i++){
				checksum[5] = checksum[5]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_sel_rows == 0){
				row = d_common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
			d_unique_d_in2_pad_cumv_sel[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_sel_elem; i++){
				checksum[6] = checksum[6]+d_unique_d_in2_pad_cumv_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_rows == 0){
				row = d_common.in2_sub_cumh_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[7] = checksum[7]+d_unique_d_in2_sub_cumh[i];
			}
		}
		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif

		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// subtract
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv_sel[ei_new] - d_unique_d_in2_sub_cumh[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[8] = checksum[8]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_rows){

			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_sub_cumh_elem; position = position + d_common.in2_sub_cumh_rows){
				d_unique_d_in2_sub_cumh[position] = d_unique_d_in2_sub_cumh[position] + sum;
				sum = d_unique_d_in2_sub_cumh[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[9] = checksum[9]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_sel_rows == 0){
				row = d_common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
			d_unique_d_in2_sub_cumh_sel[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_sel_elem; i++){
				checksum[10] = checksum[10]+d_unique_d_in2_sub_cumh_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub2_rows == 0){
				row = d_common.in2_sub2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
			d_unique_d_in2_sub2[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[11] = checksum[11]+d_unique_d_in2_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// subtract
			d_unique_d_in2_sub2[ei_new] = d_unique_d_in2_sub_cumh_sel[ei_new] - d_unique_d_in2_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[12] = checksum[12]+d_unique_d_in2_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	CUMULATIVE SUM 2
		//====================================================================================================100

		//==================================================50
		//	MULTIPLICATION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sqr_elem){

			temp = d_unique_d_in2[ei_new];
			d_unique_d_in2_sqr[ei_new] = temp * temp;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sqr_elem; i++){
				checksum[13] = checksum[13]+d_unique_d_in2_sqr[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	PAD ARRAY, VERTICAL CUMULATIVE SUM
		//==================================================50

		//==================================================50
		//	PAD ARRAY
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_elem){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_rows == 0){
				row = d_common.in2_pad_cumv_rows - 1;
				col = col-1;
			}

			// execution
			if(	row > (d_common.in2_pad_add_rows-1) &&													// do if has numbers in original array
				row < (d_common.in2_pad_add_rows+d_common.in2_sqr_rows) && 
				col > (d_common.in2_pad_add_cols-1) && 
				col < (d_common.in2_pad_add_cols+d_common.in2_sqr_cols)){
				ori_row = row - d_common.in2_pad_add_rows;
				ori_col = col - d_common.in2_pad_add_cols;
				d_unique_d_in2_pad_cumv[ei_new] = d_unique_d_in2_sqr[ori_col*d_common.in2_sqr_rows+ori_row];
			}
			else{																							// do if otherwise
				d_unique_d_in2_pad_cumv[ei_new] = 0;
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[14] = checksum[14]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	VERTICAL CUMULATIVE SUM
		//==================================================50

		//work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_cols){

			// figure out column position
			pos_ori = ei_new*d_common.in2_pad_cumv_rows;

			// variables
			sum = 0;
			
			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_pad_cumv_rows; position = position + 1){
				d_unique_d_in2_pad_cumv[position] = d_unique_d_in2_pad_cumv[position] + sum;
				sum = d_unique_d_in2_pad_cumv[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[15] = checksum[15]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_sel_rows == 0){
				row = d_common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
			d_unique_d_in2_pad_cumv_sel[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_sel_elem; i++){
				checksum[16] = checksum[16]+d_unique_d_in2_pad_cumv_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_rows == 0){
				row = d_common.in2_sub_cumh_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[17] = checksum[17]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// subtract
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv_sel[ei_new] - d_unique_d_in2_sub_cumh[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[18] = checksum[18]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_rows){

			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_sub_cumh_elem; position = position + d_common.in2_sub_cumh_rows){
				d_unique_d_in2_sub_cumh[position] = d_unique_d_in2_sub_cumh[position] + sum;
				sum = d_unique_d_in2_sub_cumh[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_rows; i++){
				checksum[19] = checksum[19]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_sel_rows == 0){
				row = d_common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
			d_unique_d_in2_sub_cumh_sel[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_sel_elem; i++){
				checksum[20] = checksum[20]+d_unique_d_in2_sub_cumh_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub2_rows == 0){
				row = d_common.in2_sub2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[21] = checksum[21]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// subtract
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sub_cumh_sel[ei_new] - d_unique_d_in2_sqr_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[22] = checksum[22]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	FINAL
		//====================================================================================================100

		//==================================================50
		//	DENOMINATOR A		SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			temp = d_unique_d_in2_sub2[ei_new];
			temp2 = d_unique_d_in2_sqr_sub2[ei_new] - (temp * temp / d_common.in_elem);
			if(temp2 < 0){
				temp2 = 0;
			}
			d_unique_d_in2_sqr_sub2[ei_new] = sqrt(temp2);
			

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[23] = checksum[23]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	MULTIPLICATION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in_sqr_elem){

			temp = d_in[ei_new];
			d_unique_d_in_sqr[ei_new] = temp * temp;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_sqr_elem; i++){
				checksum[24] = checksum[24]+d_unique_d_in_sqr[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	IN SUM
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in_cols){

			sum = 0;
			for(i = 0; i < d_common.in_rows; i++){

				sum = sum + d_in[ei_new*d_common.in_rows+i];

			}
			d_unique_d_in_partial_sum[ei_new] = sum;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_cols; i++){
				checksum[25] = checksum[25]+d_unique_d_in_partial_sum[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	IN_SQR SUM
		//==================================================50

		ei_new = tx;
		while(ei_new < d_common.in_sqr_rows){
				
			sum = 0;
			for(i = 0; i < d_common.in_sqr_cols; i++){

				sum = sum + d_unique_d_in_sqr[ei_new+d_common.in_sqr_rows*i];

			}
			d_unique_d_in_sqr_partial_sum[ei_new] = sum;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_sqr_rows; i++){
				checksum[26] = checksum[26]+d_unique_d_in_sqr_partial_sum[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	FINAL SUMMATION
		//==================================================50

		if(tx == 0){

			d_unique_d_in_final_sum[0] = 0;
			for(i = 0; i<d_common.in_cols; i++){
				// in_final_sum = in_final_sum + d_unique_d_in_partial_sum[i];
				d_unique_d_in_final_sum[0] = d_unique_d_in_final_sum[0] + d_unique_d_in_partial_sum[i];
			}

		}else if(tx == 1){

			d_unique_d_in_sqr_final_sum[0] = 0;
			for(i = 0; i<d_common.in_sqr_cols; i++){
				// in_sqr_final_sum = in_sqr_final_sum + d_unique_d_in_sqr_partial_sum[i];
				d_unique_d_in_sqr_final_sum[0] = d_unique_d_in_sqr_final_sum[0] + d_unique_d_in_sqr_partial_sum[i];
			}

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[27] = checksum[27]+d_unique_d_in_final_sum[0]+d_unique_d_in_sqr_final_sum[0];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	DENOMINATOR T
		//==================================================50

		if(tx == 0){

			// mean = in_final_sum / d_common.in_elem;													// gets mean (average) value of element in ROI
			mean = d_unique_d_in_final_sum[0] / d_common.in_elem;													// gets mean (average) value of element in ROI
			mean_sqr = mean * mean;
			// variance  = (in_sqr_final_sum / d_common.in_elem) - mean_sqr;							// gets variance of ROI
			variance  = (d_unique_d_in_sqr_final_sum[0] / d_common.in_elem) - mean_sqr;							// gets variance of ROI
			deviation = sqrt(variance);																// gets standard deviation of ROI

			// denomT = sqrt((float)(d_common.in_elem-1))*deviation;
			d_unique_d_denomT[0] = sqrt((float)(d_common.in_elem-1))*deviation;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[28] = checksum[28]+d_unique_d_denomT[i];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	DENOMINATOR		SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * denomT;
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * d_unique_d_denomT[0];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[29] = checksum[29]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	NUMERATOR	SAVE RESULT IN CONVOLUTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.conv_elem){

			// d_unique_d_conv[ei_new] = d_unique_d_conv[ei_new] - d_unique_d_in2_sub2[ei_new] * in_final_sum / d_common.in_elem;
			d_unique_d_conv[ei_new] = d_unique_d_conv[ei_new] - d_unique_d_in2_sub2[ei_new] * d_unique_d_in_final_sum[0] / d_common.in_elem;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.conv_elem; i++){
				checksum[30] = checksum[30]+d_unique_d_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	CORRELATION	SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_conv[ei_new] / d_unique_d_in2_sqr_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}



		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[31] = checksum[31]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	TEMPLATE MASK CREATE
		//====================================================================================================100

		cent = d_common.sSize + d_common.tSize + 1;
		if(frame_no == 0){
			tMask_row = cent + d_unique_d_Row[d_unique_point_no] - d_unique_d_Row[d_unique_point_no] - 1;
			tMask_col = cent + d_unique_d_Col[d_unique_point_no] - d_unique_d_Col[d_unique_point_no] - 1;
		}
		else{
			pointer = d_unique_point_no*d_common.no_frames+frame_no-1;
			tMask_row = cent + d_unique_d_tRowLoc[pointer] - d_unique_d_Row[d_unique_point_no] - 1;
			tMask_col = cent + d_unique_d_tColLoc[pointer] - d_unique_d_Col[d_unique_point_no] - 1;
		}

		//work
		ei_new = tx;
		while(ei_new < d_common.tMask_elem){

			location = tMask_col*d_common.tMask_rows + tMask_row;

			if(ei_new==location){
				d_unique_d_tMask[ei_new] = 1;
			}
			else{
				d_unique_d_tMask[ei_new] = 0;
			}

			//go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.tMask_elem; i++){
				checksum[32] = checksum[32]+d_unique_d_tMask[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	MASK CONVOLUTION
		//====================================================================================================100

		// work
		ei_new = tx;
		while(ei_new < d_common.mask_conv_elem){

			// figure out row/col location in array
			ic = (ei_new+1) % d_common.mask_conv_rows;												// (1-n)
			jc = (ei_new+1) / d_common.mask_conv_rows + 1;											// (1-n)
			if((ei_new+1) % d_common.mask_conv_rows == 0){
				ic = d_common.mask_conv_rows;
				jc = jc-1;
			}

			//
			j = jc + d_common.mask_conv_joffset;
			jp1 = j + 1;
			if(d_common.mask_cols < jp1){
				ja1 = jp1 - d_common.mask_cols;
			}
			else{
				ja1 = 1;
			}
			if(d_common.tMask_cols < j){
				ja2 = d_common.tMask_cols;
			}
			else{
				ja2 = j;
			}

			i = ic + d_common.mask_conv_ioffset;
			ip1 = i + 1;
			
			if(d_common.mask_rows < ip1){
				ia1 = ip1 - d_common.mask_rows;
			}
			else{
				ia1 = 1;
			}
			if(d_common.tMask_rows < i){
				ia2 = d_common.tMask_rows;
			}
			else{
				ia2 = i;
			}

			s = 0;

			for(ja=ja1; ja<=ja2; ja++){
				jb = jp1 - ja;
				for(ia=ia1; ia<=ia2; ia++){
					ib = ip1 - ia;
					s = s + d_unique_d_tMask[d_common.tMask_rows*(ja-1)+ia-1] * 1;
				}
			}

			// //d_unique_d_mask_conv[d_common.mask_conv_rows*(jc-1)+ic-1] = s;
			d_unique_d_mask_conv[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * s;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.mask_conv_elem; i++){
				checksum[33] = checksum[33]+d_unique_d_mask_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	MAXIMUM VALUE
		//====================================================================================================100

		//==================================================50
		//	INITIAL SEARCH
		//==================================================50

		ei_new = tx;
		while(ei_new < d_common.mask_conv_rows){

			for(i=0; i<d_common.mask_conv_cols; i++){
				largest_coordinate_current = ei_new*d_common.mask_conv_rows+i;
				largest_value_current = fabs(d_unique_d_mask_conv[largest_coordinate_current]);
				if(largest_value_current > largest_value){
					largest_coordinate = largest_coordinate_current;
					largest_value = largest_value_current;
				}
			}
			d_unique_d_par_max_coo[ei_new] = largest_coordinate;
			d_unique_d_par_max_val[ei_new] = largest_value;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.mask_conv_rows; i++){
				checksum[34] = checksum[34]+d_unique_d_par_max_coo[i]+d_unique_d_par_max_val[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	FINAL SEARCH
		//==================================================50

		if(tx == 0){

			for(i = 0; i < d_common.mask_conv_rows; i++){
				if(d_unique_d_par_max_val[i] > fin_max_val){
					fin_max_val = d_unique_d_par_max_val[i];
					fin_max_coo = d_unique_d_par_max_coo[i];
				}
			}

			// convert coordinate to row/col form
			largest_row = (fin_max_coo+1) % d_common.mask_conv_rows - 1;											// (0-n) row
			largest_col = (fin_max_coo+1) / d_common.mask_conv_rows;												// (0-n) column
			if((fin_max_coo+1) % d_common.mask_conv_rows == 0){
				largest_row = d_common.mask_conv_rows - 1;
				largest_col = largest_col - 1;
			}

			// calculate offset
			largest_row = largest_row + 1;																	// compensate to match MATLAB format (1-n)
			largest_col = largest_col + 1;																	// compensate to match MATLAB format (1-n)
			offset_row = largest_row - d_common.in_rows - (d_common.sSize - d_common.tSize);
			offset_col = largest_col - d_common.in_cols - (d_common.sSize - d_common.tSize);
			pointer = d_unique_point_no*d_common.no_frames+frame_no;
			d_unique_d_tRowLoc[pointer] = d_unique_d_Row[d_unique_point_no] + offset_row;
			d_unique_d_tColLoc[pointer] = d_unique_d_Col[d_unique_point_no] + offset_col;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[35] = checksum[35]+d_unique_d_tRowLoc[pointer]+d_unique_d_tColLoc[pointer];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	End
		//====================================================================================================100

	}

	//======================================================================================================================================================150
	//	PERIODIC COORDINATE AND TEMPLATE UPDATE
	//======================================================================================================================================================150

	if(frame_no != 0 && (frame_no)%10 == 0){


		//====================================================================================================100
		// if the last frame in the bath, update template
		//====================================================================================================100

		// update coordinate
		loc_pointer = d_unique_point_no*d_common.no_frames+frame_no;

		d_unique_d_Row[d_unique_point_no] = d_unique_d_tRowLoc[loc_pointer];
		d_unique_d_Col[d_unique_point_no] = d_unique_d_tColLoc[loc_pointer];

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = d_unique_d_Row[d_unique_point_no] - 25 + row - 1;
			ori_col = d_unique_d_Col[d_unique_point_no] - 25 + col - 1;
			ori_pointer = ori_col*d_common.frame_rows+ori_row;

			// update template
			d_in[ei_new] = d_common.alpha*d_in[ei_new] + (1-d_common.alpha)*d_frame[ori_pointer];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[36] = checksum[36]+d_in[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

	}
}
