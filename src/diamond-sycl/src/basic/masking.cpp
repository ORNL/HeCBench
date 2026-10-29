/****
  DIAMOND protein aligner
  Copyright (C) 2013-2017 Benjamin Buchfink <buchfink@gmail.com>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ****/

#include "masking.h"
#include <sycl/sycl.hpp>


#define SEQ_LEN 33
inline double firstRepeatOffsetProb(const double probMult, const int maxRepeatOffset) {
  if (probMult < 1 || probMult > 1)
    return (1 - probMult) / (1 - sycl::pow(probMult, (double)maxRepeatOffset));
  else
    return 1.0 / maxRepeatOffset;
}

void maskProbableLetters(const int size,
    unsigned char *seqBeg,
    const float *probabilities,
    const unsigned char *maskTable) {

  const double minMaskProb = 0.5;
  for (int i=0; i<size; i++)
    if (probabilities[i] >= minMaskProb)
      seqBeg[i] = maskTable[seqBeg[i]];
}

int calcRepeatProbs(float *letterProbs,
    const unsigned char *seqBeg,
    const int size,
    const int maxRepeatOffset,
    const double *likelihoodRatioMatrix, // 64 by 64 matrix,
    const double b2b,
    const double f2f0,
    const double f2b,
    const double b2fLast_inv,
    const double *pow_lkp,
    double *foregroundProbs,
    const int scaleStepSize,
    double *scaleFactors)
{

  double backgroundProb = 1.0;
  for (int k=0; k < size ; k++) {

    const int v0 = seqBeg[k];
    const int k_cap = k < maxRepeatOffset ? k : maxRepeatOffset;

    const int pad1 = k_cap - 1;
    const int pad2 = maxRepeatOffset - k_cap; // maxRepeatOffset - k, then 0                   when k > maxRepeatOffset
    const int pad3 = k - k_cap;               // 0                  , then maxRepeatOffset - k when k > maxRepeatOffset

    double accu = 0;

    for (int i = 0; i < k; i++) {

      const int idx1 = pad1 - i;
      const int idx2 = pad2 + i;
      const int idx3 = pad3 + i;

      const int v1 = seqBeg[idx3];
      accu += foregroundProbs[idx1];
      foregroundProbs[idx1] = ( (f2f0 * foregroundProbs[idx1]) +
          (backgroundProb * pow_lkp[idx2]) ) *
        likelihoodRatioMatrix[v0*size+v1];
    }

    backgroundProb = (backgroundProb * b2b) + (accu * f2b);

    if (k % scaleStepSize == scaleStepSize - 1) {
      const double scale = 1 / backgroundProb;
      scaleFactors[k / scaleStepSize] = scale;

      for (int i=0; i< k_cap; i++)
        foregroundProbs[i] = foregroundProbs[i] * scale;

      backgroundProb = 1;
    }

    letterProbs[k] = (float)(backgroundProb);
  }

  double accu = 0;
  for (int i=0 ; i < maxRepeatOffset; i++) {
    accu += foregroundProbs[i];
    foregroundProbs[i] = f2b;
  }

  const double fTot = backgroundProb * b2b + accu * f2b;
  backgroundProb = b2b;

  const double fTot_inv = 1/ fTot ;
  for (int k=(size-1) ; k >= 0 ; k--){


    float nonRepeatProb = letterProbs[k] * backgroundProb * fTot_inv;
    letterProbs[k] = 1.f - nonRepeatProb;

    const int k_cap  = sycl::min(k, maxRepeatOffset);

    if (k % scaleStepSize == scaleStepSize - 1) {
      const double scale = scaleFactors[k/ scaleStepSize];

      for (int i=0; i< k_cap; i++)
        foregroundProbs[i] = foregroundProbs[i] * scale;

      backgroundProb *= scale;
    }

    const double c0 = f2b * backgroundProb;
    const int v0= seqBeg[k];

    double accu = 0;
    for (int i = 0; i < k_cap; i++) {


      const int v1 =  seqBeg[k-(i+1)];
      const double f = foregroundProbs[i] * likelihoodRatioMatrix[v0*size+v1];

      accu += pow_lkp[k_cap-(i+1)]*f;
      foregroundProbs[i] = c0 + f2f0 * f;
    }

    const double p = k > maxRepeatOffset ? 1. : pow_lkp[maxRepeatOffset - k]*b2fLast_inv;
    backgroundProb = (b2b * backgroundProb) + accu*p;
  }

  const double bTot = backgroundProb;
  return (sycl::fabs(fTot - bTot) > sycl::fmax(fTot, bTot) / 1e6);
}

void maskSequences(
    unsigned char * seqs,
    const double * likelihoodRatioMatrix,
    const unsigned char * maskTable,
    const int size                     ,
    const int maxRepeatOffset          ,
    const double repeatProb            ,
    const double repeatEndProb         ,
    const double repeatOffsetProbDecay ,
    const double firstGapProb          ,
    const double otherGapProb          ,
    const double minMaskProb           ,
    int seqs_len,
    sycl::nd_item<1> &item )
{
  int gid = item.get_global_id(0);
  if (gid >= seqs_len) return;

  unsigned char* seqBeg = seqs+gid*33;

  float probabilities[SEQ_LEN];

  const double b2b = 1 - repeatProb;
  const double f2f0 = 1 - repeatEndProb;
  const double f2b = repeatEndProb;

  const double b2fGrowth = 1 / repeatOffsetProbDecay;

  const double  b2fLast = repeatProb * firstRepeatOffsetProb(b2fGrowth, maxRepeatOffset);
  const double b2fLast_inv = 1 / b2fLast ;

  double p = b2fLast;
  double ar_1[50];

  for (int i=0 ; i < maxRepeatOffset; i++){
    ar_1[i] = p ;
    p *= b2fGrowth;
  }

  const int scaleStepSize = 16;

  double scaleFactors[SEQ_LEN / scaleStepSize];

  double foregroundProbs[50];

  for (int i=0 ; i < maxRepeatOffset; i++){
    foregroundProbs[i] = 0;
  };

  const int err  = calcRepeatProbs(probabilities,seqBeg, size,
      maxRepeatOffset, likelihoodRatioMatrix,
      b2b, f2f0, f2b,
      b2fLast_inv,ar_1,foregroundProbs,scaleStepSize, scaleFactors);

  //if (err)  printf("tantan: warning: possible numeric inaccuracy\n");

  maskProbableLetters(size,seqBeg, probabilities, maskTable);
}


unique_ptr<Masking> Masking::instance;
const uint8_t Masking::bit_mask = 128;

Masking::Masking(const Score_matrix &score_matrix)
{
  const double lambda = score_matrix.lambda(); // 0.324032
  for (unsigned i = 0; i < size; ++i) {
    mask_table_x_[i] = value_traits.mask_char;
    mask_table_bit_[i] = (uint8_t)i | bit_mask;
    for (unsigned j = 0; j < size; ++j)
      if (i < value_traits.alphabet_size && j < value_traits.alphabet_size)
        likelihoodRatioMatrix_[i][j] = std::exp(lambda * score_matrix(i, j));
  }
  std::copy(likelihoodRatioMatrix_, likelihoodRatioMatrix_ + size, probMatrixPointers_);
  int firstGapCost = score_matrix.gap_extend() + score_matrix.gap_open();
  firstGapProb_ = std::exp(-lambda * firstGapCost);
  otherGapProb_ = std::exp(-lambda * score_matrix.gap_extend());
  firstGapProb_ /= (1 - otherGapProb_);
}

void Masking::operator()(Letter *seq, size_t len) const
{

  tantan::maskSequences((tantan::uchar*)seq, (tantan::uchar*)(seq + len), 50,
      (tantan::const_double_ptr*)probMatrixPointers_,
      0.005, 0.05,
      0.9,
      0, 0,
      0.5, (const tantan::uchar*)mask_table_x_);
}

unsigned char* Masking::call_opt(Sequence_set &seqs) const
{
  const int n = seqs.get_length();
  int total = 0;
  for (int i=0; i < n; i++)
    total += seqs.length(i);

  printf("There are %d sequences and the total sequence length is %d\n", n, total);
  unsigned char *seqs_device = NULL;
  posix_memalign((void**)&seqs_device, 1024, total);

  unsigned char *p = seqs_device;
  for (int i=0; i < n; i++) {
    memcpy(p, seqs.ptr(i), seqs.length(i));
    p += seqs.length(i);
  }

  double *probMat_device = NULL;
  posix_memalign((void**)&probMat_device, 1024, size*size*sizeof(double));
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      probMat_device[i*size+j] = probMatrixPointers_[i][j];

  unsigned char *mask_table_device = NULL;
  posix_memalign((void**)&mask_table_device, 1024, size*sizeof(unsigned char));
  for (int i = 0; i < size; i++)
    mask_table_device[i] = mask_table_x_[i];

  int len = 33;

  printf("Timing the mask sequences on device...\n");
  Timer t;
  t.start();

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int BLOCK_SIZE=128;
  sycl::range<1> gws ((n+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  const int size = len;
  const int maxRepeatOffset = 50;
  const double repeatProb = 0.005;
  const double repeatEndProb = 0.05;
  const double repeatOffsetProbDecay = 0.9;
  const double firstGapProb = 0;
  const double otherGapProb = 0;
  const double minMaskProb = 0.5;
  const int seqs_len = n;

  unsigned char *d_seqs = (unsigned char*) sycl::malloc_device(total, q);
  q.memcpy(d_seqs, seqs_device, total);

  unsigned char *d_maskTable = sycl::malloc_device<unsigned char>(size, q);
  q.memcpy(d_maskTable, mask_table_device, size);

  double *d_probMat = sycl::malloc_device<double>(size*size, q);
  q.memcpy(d_probMat, probMat_device, sizeof(double)*size*size);

  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
     maskSequences(
         d_seqs,
         d_probMat,
         d_maskTable,
         size,
         maxRepeatOffset,
         repeatProb,
         repeatEndProb,
         repeatOffsetProbDecay,
         firstGapProb,
         otherGapProb,
         minMaskProb,
         seqs_len,
         item);
     });
  });

  q.memcpy(seqs_device, d_seqs, total).wait();
  sycl::free(d_seqs, q);
  sycl::free(d_maskTable, q);
  sycl::free(d_probMat, q);

  message_stream << "Total time (maskSequences) on the device = " <<
    t.getElapsedTimeInMicroSec() / 1e6 << " s" << std::endl;

  free(probMat_device);
  free(mask_table_device);
  return seqs_device;
}

void Masking::call_opt(Letter *seq, size_t len) const
{
  // CPU
  tantale::maskSequences((tantan::uchar*)seq, (tantan::uchar*)(seq + len), 50,
      (tantan::const_double_ptr*)probMatrixPointers_,
      0.005, 0.05,
      0.9,
      0, 0,
      0.5, (const tantan::uchar*)mask_table_x_);
}



void Masking::mask_bit(Letter *seq, size_t len) const
{

  tantan::maskSequences((tantan::uchar*)seq, (tantan::uchar*)(seq + len), 50,
      (tantan::const_double_ptr*)probMatrixPointers_,
      0.005, 0.05,
      0.9,
      0, 0,
      0.5,		(const tantan::uchar*)mask_table_bit_);
}

void Masking::bit_to_hard_mask(Letter *seq, size_t len, size_t &n) const
{
  for (size_t i = 0; i < len; ++i)
    if (seq[i] & bit_mask) {
      seq[i] = value_traits.mask_char;
      ++n;
    }
}

void Masking::remove_bit_mask(Letter *seq, size_t len) const
{
  for (size_t i = 0; i < len; ++i)
    if (seq[i] & bit_mask)
      seq[i] &= ~bit_mask;
}

void mask_worker(Atomic<size_t> *next, Sequence_set *seqs, const Masking *masking, bool hard_mask)
{
  size_t i;
  //int cnt = 0;

  while ((i = (*next)++) < seqs->get_length())
  {
    if (hard_mask)
      //masking->operator()(seqs->ptr(i), seqs->length(i));
      masking->call_opt(seqs->ptr(i), seqs->length(i));
    else
      masking->mask_bit(seqs->ptr(i), seqs->length(i));
    //cnt++;
    //if (cnt == 2) break;
  }
}

void mask_seqs(Sequence_set &seqs, const Masking &masking, bool hard_mask)
{

  assert(hard_mask==true);
  const int n = seqs.get_length();

  printf("Timing the mask sequences on CPU...\n");
  Timer total;
  total.start();

#if not defined(_OPENMP)
  Thread_pool threads;
  Atomic<size_t> next(0);
  for (size_t i = 0; i < config.threads_; ++i)
    threads.push_back(launch_thread(mask_worker, &next, &seqs, &masking, hard_mask));
  threads.join_all();

#else

#pragma omp parallel for num_threads(config.threads_)
  for (int i=0; i < n; i++){
    masking.call_opt(seqs.ptr(i), seqs.length(i));
  }

#endif

  message_stream << "Total time (maskSequences) on the CPU = " <<
    total.getElapsedTimeInMicroSec() / 1e6 << " s" << std::endl;

  // on the device
  unsigned char* seqs_device = masking.call_opt(seqs);

  printf("Verify the sequences...\n");
  unsigned char* p = seqs_device;
  int error = 0;
  for (int i = 0; i < n; i++) {
    if (0 != strncmp((const char*)p, seqs.ptr(i), seqs.length(i))) {
      printf("error at i=%d  length=%zu\n", i, seqs.length(i));
      printf("host=");
      char* s = seqs.ptr(i);
      for (int j = 0; j < seqs.length(i); j++) {
        printf("%02d", s[j]);
      }
      printf("\ndevice=");
      for (int j = 0; j < seqs.length(i); j++)
        printf("%02d", *(seqs_device+i*33+j));
      printf("\n");
      error++;
    }
    p += seqs.length(i);
  }
  free(seqs_device);
  if (error == 0) printf("Success\n");
}
