#ifndef _ORDERGRAPH_KERNEL_H_
#define _ORDERGRAPH_KERNEL_H_
#include <stdio.h>
#include "data45.h"

void Dincr(int *bit,int n);
void DincrS(int *bit,int n);
bool D_getState(int parN,int *sta,int time);
void D_findComb(int* comb, int l, int n);
int D_findindex(int *arr, int size);
int D_C(int n, int a);


void genScoreKernel(const int sizepernode,
                    float *D_localscore, 
                    const int *D_data,
                    const float *D_LG,
                    sycl::nd_item<1> &item)
{
  int id = item.get_global_id(0);
  int node,index;
  bool flag;
  int parent[5]={0};
  int pre[NODE_N]={0};
  int state[5]={0};
  int i,j,parN=0,tmp,t;
  int t1=0,t2=0;
  float ls=0;
  int Nij[STATE_N]={0};

  if(id<sizepernode){

    D_findComb(parent,id,NODE_N-1);

    for(i=0;i<4;i++)
    {
      if(parent[i]>0) parN++;
    }

    for(node=0;node<NODE_N;node++){

      j=1;
      for(i=0;i<NODE_N;i++)
      {
        if(i!=node)pre[j++]=i;

      }

      for(tmp=0;tmp<parN;tmp++)
        state[tmp]=0;

      index=sizepernode*node+id;

      t=0;
      while(D_getState(parN,state,t++)){   // for get state
        ls=0;
        for(tmp=0;tmp<STATE_N;tmp++)
          Nij[tmp]=0;

        for(t1=0;t1<DATA_N;t1++){
          flag=true;
          for(t2=0;t2<parN;t2++){
            if(D_data[t1*NODE_N+pre[parent[t2]]]!=state[t2]) {
              flag=false;
              break;
            }
          }
          if(!flag) continue;

          Nij[D_data[t1*NODE_N+node]]++;

        }

        tmp=STATE_N-1;

        for(t1=0;t1<STATE_N;t1++){
          ls+=D_LG[Nij[t1]];
          tmp+=Nij[t1];
        }

        ls-=D_LG[tmp];
        ls+=D_LG[STATE_N-1];

        D_localscore[index]+=ls;
      }
    }
  }
}

void computeKernel(const int taskperthr,
                   const int sizepernode, 
                   const float *D_localscore, 
                   const bool *D_parent, 
                   const int node, 
                   const int total, 
                   float *D_Score,
                   int *D_resP,
                   float* lsinblock,
                   sycl::nd_item<1> &item)
{
  const int tid = item.get_local_id(0);
  const int bid = item.get_group(0);
  const int id = bid * 256 + tid;

  int posN=1,i,index,t,tmp;
  int pre[NODE_N]={0};
  int parN=0;
  int bestparent[4]={0},parent[5]={-1};
  float bestls=-999999999999999.f,ls;

  for(i=0;i<NODE_N;i++){
    if(D_parent[i]==1){pre[posN++]=i;}
  }

  for(i=0;i<taskperthr&&((id*taskperthr+i)<total);i++){

    D_findComb(parent,id*taskperthr+i,posN-1);

    for(parN=0;parN<4;parN++){
      if(parent[parN]<0) break;
      if(pre[parent[parN]]>node) parent[parN]=pre[parent[parN]];
      else                       parent[parN]=pre[parent[parN]]+1;
    }

    for(tmp=parN;tmp>0;tmp--){
      parent[tmp]=parent[tmp-1];
    }
    parent[0]=0;

    index=D_findindex(parent,parN);
    index+=sizepernode*node;

    ls=D_localscore[index];

    if(ls>bestls){
      bestls=ls;
      for(tmp=0;tmp<4;tmp++)
        bestparent[tmp]=parent[tmp+1];
    }
  }

  lsinblock[tid]=bestls;

  item.barrier(sycl::access::fence_space::local_space);

  for(i=128;i>=1;i/=2){

    if(tid<i){
      if(lsinblock[tid+i]>lsinblock[tid]&&lsinblock[tid+i]<0){
        lsinblock[tid]=lsinblock[tid+i];
        lsinblock[tid+i]=(float)(tid+i);
      }
      else if(lsinblock[tid+i]<lsinblock[tid]&&lsinblock[tid]<0){
        lsinblock[tid+i]=(float)tid;
      }
      else if(lsinblock[tid]>0&&lsinblock[tid+i]<0){
        lsinblock[tid]=lsinblock[tid+i];
        lsinblock[tid+i]=(float)(tid+i);
      }
      else if(lsinblock[tid]<0&&lsinblock[tid+i]>0){
        lsinblock[tid+i]=(float)tid;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  if(tid==0){
    D_Score[bid]=lsinblock[0];
    t=0;
    for(i=0;i<7&&t<128&&t>=0;i++){
      t=(int)lsinblock[(int)sycl::powr(2.f,(float)i)+t];
    }
    lsinblock[0]=(float)t;
  }

  item.barrier(sycl::access::fence_space::local_space);

  if(tid==(int)lsinblock[0]){
    for(i=0;i<4;i++){
      D_resP[bid*4+i]=bestparent[i];
    }
  }
}

void Dincr(int *bit,int n){

  while(n<=NODE_N){
    bit[n]++;
    if(bit[n]>=2)
    {
      bit[n]=0;
      n++;
    }
    else{
      break;
    }
  }

  return;
}

void DincrS(int *bit,int n){

  bit[n]++;
  if(bit[n]>=STATE_N)
  {
    bit[n]=0;
    Dincr(bit,n+1);
  }

  return;
}

bool D_getState(int parN,int *sta,int time){
  int i,j=1;

  for(i=0;i<parN;i++){
    j*=STATE_N;
  }
  j--;
  if(time>j) return false;

  if(time>=1)
    DincrS(sta,0);

  return true;

}

void D_findComb(int* comb, int l, int n)
{
  const int len = 4;
  if (l == 0)
  {
    for (int i = 0; i < len; i++)
      comb[i] = -1;
    return;
  }
  int sum = 0;
  int k = 1;

  while (sum < l)
    sum += D_C(n,k++);
  l -= sum - D_C(n,--k);
  int low = 0;
  int pos = 0;
  while (k > 1)
  {
    sum = 0;
    int s = 1;
    while (sum < l)
      sum += D_C(n-s++,k-1);
    l -= sum - D_C(n-(--s),--k);
    low += s;
    comb[pos++] = low;
    n -= s;
  }
  comb[pos] = low + l;
  for (int i = pos+1; i < 4; i++)
    comb[i] = -1;
}

int D_findindex(int *arr, int size){  //reminder: arr[0] has to be 0 && size == array size-1 && index start from 0
  if (size == 0) return 0;

  int i,j,index=0;

  for(i=1;i<size;i++){
    index+=D_C(NODE_N-1,i);
  }

  for(i=1;i<=size-1;i++){
    for(j=arr[i-1]+1;j<=arr[i]-1;j++){
      index+=D_C(NODE_N-1-j,size-i);
    }
  }

  index+=arr[size]-arr[size-1];

  return index;

}

int D_C(int n, int a){
  int i,res=1,atmp=a;

  for(i=0;i<atmp;i++){
    res*=n;
    n--;
  }

  for(i=0;i<atmp;i++){
    res/=a;
    a--;
  }

  return res;
}

#endif
