/*
 Condition-dependent Correlation Subgroups (CCS) 
 Description: Biclustering has been emerged as a powerful tool for 
 identification of a group of co-expressed genes under a subset 
 of experimental conditions (measurements) present in a gene 
 expression dataset.  In this program we implemented CCS biclustering. 

 Developer: Dr. Anindya Bhattacharya and Dr. Yan Cui, UTHSC, Memphis, TN, USA
 Email: anindyamail123@gmail.com; ycui2@uthsc.edu 

 Note: The minimum number of genes and the samples per bicluster is 10. 
 User can alter the minimum size by changing the values for 'mingene' 
 and 'minsample' defined in "ccs.h" file for minimum number of genes and samples
 respectively. 
*/

#include <chrono>
#include <cuda.h>
#include "ccs.h"
#include "matrixsize.c"
#include "readgene.c"
#include "pair_cor.c"
#include "bicluster_pair_score.c"
#include "merge_bicluster.c"
#include "print_bicluster.c"

// number of samples in the input datamatrix. 
// Fixed here to make static shared memory on a device
#define MAXSAMPLE 200 

__device__
struct pair_r compute(
  float *genekj,
  float *geneij,
  const char *sample,
  int wid,int k,int i,int D,
  const float *gene)
{
  int j;
  float sx = 0.f, sxx = 0.f, sy = 0.f, sxy = 0.f, syy = 0.f;
  float sx_n = 0.f, sxx_n = 0.f, sy_n = 0.f, sxy_n = 0.f, syy_n = 0.f;

  struct pair_r rval = {0.f, 0.f};

  for (j = 0; j < D; j++) {
    genekj[j]=gene[k*(D+1)+j];
    if(sample[j]=='1')
      sx += genekj[j];
    else
      sx_n += genekj[j];
  }

  sx /= wid;
  sx_n /= (D-wid);

  for (j = 0; j < D; j++) {
    if(sample[j]=='1')
      sxx += (sx-genekj[j]) * (sx-genekj[j]);
    else
      sxx_n += (sx_n-genekj[j]) * (sx_n-genekj[j]);
  }

  sxx = sqrtf(sxx);
  sxx_n = sqrtf(sxx_n);

  for (j = 0; j < D; j++) {
    geneij[j]=gene[i*(D+1)+j];
    if(sample[j]=='1')
      sy += geneij[j];
    else
      sy_n += geneij[j];
  }

  sy /= wid; 
  sy_n /= (D-wid); 

  for (j = 0; j < D; j++)
  {
    if(sample[j]=='1') {
      sxy += (sx - genekj[j]) * (sy - geneij[j]);
      syy += (sy - geneij[j]) * (sy - geneij[j]);
    }
    else {
      sxy_n += (sx_n - genekj[j]) * (sy_n - geneij[j]);
      syy_n += (sy_n - geneij[j]) * (sy_n - geneij[j]);
    }
  }

  syy = sqrtf(syy);
  syy_n = sqrtf(syy_n);
  rval.r = fabsf(sxy/(sxx * syy));
  rval.n_r = fabsf(sxy_n/(sxx_n * syy_n));

  return rval;
}

__global__ void compute_bicluster(
  const float *__restrict__ gene, 
  const int n,
  const int maxbcn,
  const int D,
  const float thr,
  char *__restrict__ maxbc_sample,
  char *__restrict__ maxbc_data,
  float *__restrict__ maxbc_score,
  int *__restrict__ maxbc_datacount,
  int *__restrict__ maxbc_samplecount,
  char *__restrict__ tmpbc_sample,
  char *__restrict__ tmpbc_data)
{
  __shared__ float s_genekj[MAXSAMPLE];
  __shared__ float s_geneij[MAXSAMPLE];
  __shared__  char s_vect[3*MAXSAMPLE];

  int k=blockIdx.x*blockDim.x+threadIdx.x;

  if(k<maxbcn) {
    float jcc,mean_k,mean_i;
    int i,j,l,vl,wid,wid_0,wid_1,wid_2,l_i,t_tot,t_dif;
    int dif,tot;
    struct pair_r rval;
    int tmpbc_datacount,tmpbc_samplecount;

    float genekj,geneij;

    maxbc_score[k]=1.f;
    maxbc_datacount[k]=0;  

    //calculate mean expression for gene k

    mean_k=gene[k*(D+1)+D];

    for (i = k+1; i < n; i++) //pair k,i
    {   
      //calculate mean expression for gene i
      mean_i=gene[i*(D+1)+D];

      wid_0=0; wid_1=0; wid_2=0;      

      for (j = 0; j < D; j++)  
      {
        genekj=gene[k*(D+1)+j];
        geneij=gene[i*(D+1)+j];

        if ((genekj - mean_k)>=0 && (geneij - mean_i)>=0) //i and k upregulated : positive correlation
        {
          s_vect[0*MAXSAMPLE+j] = '1';
          s_vect[1*MAXSAMPLE+j] = '0';
          s_vect[2*MAXSAMPLE+j] = '0';
          wid_0++;
        }
        else if ((genekj - mean_k)<0 && (geneij - mean_i)<0)  // i and k down regulated : positive correlation
        {
          s_vect[0*MAXSAMPLE+j] = '0';
          s_vect[1*MAXSAMPLE+j] = '1';
          s_vect[2*MAXSAMPLE+j] = '0';
          wid_1++;
        }
        else if ((genekj - mean_k)*(geneij - mean_i)<0) //betwenn i and k one is up regulated and the other one is down regulated : negative correlation
        {
          s_vect[0*MAXSAMPLE+j] = '0';
          s_vect[1*MAXSAMPLE+j] = '0';
          s_vect[2*MAXSAMPLE+j] = '1';
          wid_2++;
        } 
      }

      for (vl = 0; vl < 3; vl++)
      { 
        dif=0; tot=0;
        if(vl==0)
          wid=wid_0; 
        else if(vl==1)
          wid=wid_1; 
        if(vl==2)
          wid=wid_2; 

        if(wid>minsample) { //minimum samples required to form a bicluster module. Default minimum set to 10 in ccs.h   

          rval=compute(s_genekj, s_geneij, s_vect+vl*MAXSAMPLE, wid, k, i, D, gene);
        }
        else {
          continue;
        }

        if (rval.r > thr) 
        {
          tot++;      
          if(rval.n_r>thr)
            dif++;

          for (j = 0;j < D; j++)
            tmpbc_sample[k*D+j] = s_vect[vl*MAXSAMPLE+j];

          for (j = 0;j < n; j++)
            tmpbc_data[k*n+j] = '0';

          tmpbc_data[k*n+k] = '1';
          tmpbc_data[k*n+i] = '1';
          tmpbc_datacount = 2;
          tmpbc_samplecount = wid;

          for (l = 0; l < n; l++)  { //bicluster augmentation
            if (l != i && l != k) {
              t_tot=0; t_dif=0;
              for(l_i=0;l_i<n;l_i++) {
                if(tmpbc_data[k*n+l_i]=='1')  {
                  rval=compute(s_genekj, s_geneij, s_vect + vl*MAXSAMPLE, wid, l, l_i, D, gene);

                  if(rval.r>thr) 
                    t_tot+=1;
                  else {
                    t_tot=0;
                    break;
                  }   
                  if(rval.n_r>thr) 
                    t_dif+=1;
                }  
              }                                                                    

              if(t_tot>0)  {
                tmpbc_data[k*n+l] = '1';
                tmpbc_datacount+=1;
                tot+=t_tot; dif+=t_dif;
              }
            }
          }  // end of augmentation

          // Compute Jaccard score

          if(tot>0)
            jcc=(float)dif/tot;   
          else
            jcc=1.f; 

          /* Select bicluster candidate as the largest (maxbc[k].datacount<tmpbc.datacount) 
             of all condition dependent (jaccard score <0.01) bicluster for k. Minimum number of gene 
             for a bicluster is set at 10. See the mingene at ccs.h */

          if(jcc<0.01f && maxbc_datacount[k]<tmpbc_datacount && tmpbc_datacount>mingene)
          {
            maxbc_score[k]=jcc;
            for (j = 0; j < n; j++)  
              maxbc_data[k*n+j]=tmpbc_data[k*n+j];
            for (j = 0; j < D; j++)  
              maxbc_sample[k*D+j]=tmpbc_sample[k*D+j];
            maxbc_datacount[k]=tmpbc_datacount;
            maxbc_samplecount[k]=tmpbc_samplecount;
          }
        }    //end of r>thr condition
      }    //end of loop for vl  
    }  // end of i loop
  }
}

int main(int argc, char *argv[])
{
  FILE *in,*out;
  struct gn *gene;
  char **Hd;
  char *infile,*outfile;  
  int c, errflag;
  int maxbcn=MAXB;
  int print_type=0;
  int repeat=0;
  int i,n,D;
  extern char *optarg;
  float thr;
  struct bicl *bicluster;
  float overlap=100.f; 

  infile = outfile = NULL;
  in = out = NULL;

  errflag = n = D = 0;
  thr = 0.f;

  while ((c = getopt(argc, argv, "ht:m:r:i:p:o:g:?")) != -1)
  {
    switch(c)
    {
      case 'h': // help
        printUsage();
        exit(0);
      case 't': // threshold value
        thr = atof(optarg);
        break;
      case 'm': // maximum number of bicluster search
        maxbcn = atoi(optarg);
        break;
      case 'r': // kernel repeat times
        repeat = atoi(optarg);
        break;
      case 'g': // output file format
        overlap = atof(optarg);
        break;
      case 'p': // output file format
        print_type = atoi(optarg);
        break;
      case 'i': // the input expression file
        infile = optarg;
        break;
      case 'o': // the output file
        outfile = optarg;
        break;
      case ':':       /* -f or -o without operand */
        printf("Option -%c requires an operand\n", optopt);
        errflag++;
        break;
      case '?':
        fprintf(stderr,"Unrecognized option: -%c\n", optopt);
        errflag++;
    }
  }

  if (thr == 0)
  {
    fprintf(stderr,"***** WARNING: Threshold Theta (corr coeff) "
        "value assumed to be ZERO (0)\n");
  }

  if (outfile == NULL)
  {
    fprintf(stderr,"***** WARNING: Output file assumed to be STDOUT\n");
    out = stdout;
  }
  else if ((out = fopen(outfile,"w")) == NULL) //write open bicluster file
  {
    fprintf(stderr,"***** ERROR: Unable to open Output file %s\n",outfile);
    errflag++;
  }

  if ((thr < 0) || (thr > 1))
  {
    fprintf(stderr,"***** ERROR: Threshold Theta (corr coeff) "
        "must be between 0.0-1.0\n");
  }

  if (infile == NULL)
  {
    fprintf(stderr,"***** ERROR: Input file not defined\n");
    if (out) fclose(out);
    errflag++;
  }
  else if ((in = fopen(infile,"r")) == NULL)  //open gene file
  {
    fprintf(stderr,"***** ERROR: Unable to open Input %s\n", infile);
    if (out) fclose(out);
    errflag++;
  }

  if (errflag)
  {
    printUsage();
    exit(1);
  }

  getmatrixsize(in,&n,&D);
  printf("Number of rows=%d\tNumber of columns=%d\n",n,D);

  if(maxbcn>n) maxbcn=n;

  gene = (struct gn *)calloc(n,sizeof(struct gn));
  Hd = (char **)calloc(D+1,sizeof(char *));

  for (i = 0; i < n; i++)
    gene[i].x = ( float *)calloc(D+1,sizeof( float));

  bicluster = (struct bicl *)calloc(maxbcn,sizeof(struct bicl));
  for (i = 0; i < maxbcn; i++)
  {
    bicluster[i].sample = (char *)calloc(D,sizeof(char));
    bicluster[i].data = (char *)calloc(n,sizeof(char));
  }

  // initialize the gene data
  readgene(infile,gene,Hd,n,D);  

  auto start = std::chrono::steady_clock::now();

  float *d_gene;
  cudaMalloc((void**)&d_gene, sizeof(float) * n * (D+1));

  for (i = 0; i < n; i++) {
    cudaMemcpy(d_gene+i*(D+1), gene[i].x, sizeof(float)*(D+1), cudaMemcpyHostToDevice);  
  }

  float *d_bc_score;
  cudaMalloc((void**)&d_bc_score, sizeof(float)*maxbcn);

  int *d_bc_datacount;
  cudaMalloc((void**)&d_bc_datacount, sizeof(int)*maxbcn);

  int *d_bc_samplecount;
  cudaMalloc((void**)&d_bc_samplecount, sizeof(int)*maxbcn);

  char *d_bc_sample;
  cudaMalloc((void**)&d_bc_sample, sizeof(char)*D*maxbcn);

  char *d_bc_sample_tmp;
  cudaMalloc((void**)&d_bc_sample_tmp, sizeof(char)*D*maxbcn);

  char *d_bc_data;
  cudaMalloc((void**)&d_bc_data, sizeof(char)*n*maxbcn);

  char *d_bc_data_tmp;
  cudaMalloc((void**)&d_bc_data_tmp, sizeof(char)*n*maxbcn);

  dim3 blocks (1);
  dim3 grids (maxbcn);

  cudaDeviceSynchronize();
  auto kstart = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    compute_bicluster <<< grids, blocks >>> (
      d_gene,
      n,maxbcn,D,thr,
      d_bc_sample,
      d_bc_data,
      d_bc_score,
      d_bc_datacount,
      d_bc_samplecount,
      d_bc_sample_tmp,
      d_bc_data_tmp);
  }

  cudaDeviceSynchronize();
  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  printf("Average kernel execution time %f (s)\n", ktime * 1e-9f / repeat);

  float *bicluster_temp_score = (float *)calloc(maxbcn,sizeof(float));
  cudaMemcpy(bicluster_temp_score, d_bc_score, sizeof(float)*maxbcn, cudaMemcpyDeviceToHost);

  int *bicluster_temp_datacount = (int *)calloc(maxbcn,sizeof(int));
  cudaMemcpy(bicluster_temp_datacount, d_bc_datacount, sizeof(int)*maxbcn, cudaMemcpyDeviceToHost);

  int *bicluster_temp_samplecount = (int *)calloc(maxbcn,sizeof(int));
  cudaMemcpy(bicluster_temp_samplecount, d_bc_samplecount, sizeof(int)*maxbcn, cudaMemcpyDeviceToHost);

  for(i=0; i<maxbcn; i++) {
    cudaMemcpy(bicluster[i].sample, d_bc_sample_tmp+D*i, sizeof(char)*D, cudaMemcpyDeviceToHost);
    cudaMemcpy(bicluster[i].data, d_bc_data_tmp+n*i, sizeof(char)*n, cudaMemcpyDeviceToHost);
  }

  cudaFree(d_gene);
  cudaFree(d_bc_score);
  cudaFree(d_bc_datacount);
  cudaFree(d_bc_samplecount);
  cudaFree(d_bc_sample);
  cudaFree(d_bc_sample_tmp);
  cudaFree(d_bc_data);
  cudaFree(d_bc_data_tmp);

  for(i=0; i<maxbcn; i++) {
    bicluster[i].score=bicluster_temp_score[i];
    bicluster[i].datacount=bicluster_temp_datacount[i];
    bicluster[i].samplecount=bicluster_temp_samplecount[i];
  }

  printbicluster(out,gene,Hd,n,D,maxbcn,thr,bicluster,print_type,overlap);

  for (i = 0; i < n; i++) {
    free(gene[i].x);
    free(gene[i].id);
  }
  free(gene);
  for (i = 0; i < D+1; i++) free(Hd[i]);
  free(Hd);

  for (i = 0; i < maxbcn; i++) {  
    free(bicluster[i].sample);
    free(bicluster[i].data);
  }

  free(bicluster_temp_score);
  free(bicluster_temp_datacount);
  free(bicluster_temp_samplecount);
  free(bicluster);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Elapsed time = %f (s)\n", time * 1e-9f);
  if (print_type==0) fprintf(out,"\n\nElapsed time = %f s\n", time * 1e-9f);
  if (out) fclose(out);

  return 0;
}
