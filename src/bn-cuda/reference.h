const int HIGHEST = 3;
const int ITER = 100;
const int WORKLOAD = 1;
int sizepernode;

float preScore = -99999999999.f;
float score;
float maxScore[HIGHEST] = {-999999999.f};
bool orders[NODE_N][NODE_N];
bool preOrders[NODE_N][NODE_N];
bool preGraph[NODE_N][NODE_N];
bool bestGraph[HIGHEST][NODE_N][NODE_N];
bool graph[NODE_N][NODE_N];
float *localscore, *scores;
float *LG;
int *parents;
float score_ref;
float *localscore_ref;

void initial();  // initial orders and data
int genOrders(); // swap
int ConCore();   // discard new order or not
// get every possible set of parents for a node
void incr(int *bit, int n);  // binary code increases 1 each time
void incrS(int *bit, int n); // STATE_N code increases 1 each time
// get every possible combination of state for a parent set
bool getState( int parN, int *state, int time);
float logGamma(int N); // log and gamma
void genScore();
void sortGraph();
void swap(int a, int b);
void Pre_logGamma();
int findindex(int *arr, int size);
int C(int n, int a);

void Hincr(int *bit,int n);
void HincrS(int *bit,int n);
bool H_getState(int parN,int *sta,int time);
void H_findComb(int* comb, int l, int n);
int H_findindex(int *arr, int size);
int H_C(int n, int a);

void Hincr(int *bit,int n){

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

void HincrS(int *bit,int n){

  bit[n]++;
  if(bit[n]>=STATE_N)
  {
    bit[n]=0;
    Hincr(bit,n+1);
  }

  return;
}

bool H_getState(int parN,int *sta,int time){
  int i,j=1;

  for(i=0;i<parN;i++){
    j*=STATE_N;
  }
  j--;
  if(time>j) return false;

  if(time>=1)
    HincrS(sta,0);

  return true;

}

void H_findComb(int* comb, int l, int n)
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
    sum += H_C(n,k++);
  l -= sum - H_C(n,--k);
  int low = 0;
  int pos = 0;
  while (k > 1)
  {
    sum = 0;
    int s = 1;
    while (sum < l)
      sum += H_C(n-s++,k-1);
    l -= sum - H_C(n-(--s),--k);
    low += s;
    comb[pos++] = low;
    n -= s;
  }
  comb[pos] = low + l;
  for (int i = pos+1; i < 4; i++)
    comb[i] = -1;
}

int H_findindex(int *arr, int size){  //reminder: arr[0] has to be 0 && size == array size-1 && index start from 0
  if (size == 0) return 0; // empty parent set maps to slot 0 (matches H_findComb(l=0))

  int i,j,index=0;

  for(i=1;i<size;i++){
    index+=H_C(NODE_N-1,i);
  }

  for(i=1;i<=size-1;i++){
    for(j=arr[i-1]+1;j<=arr[i]-1;j++){
      index+=H_C(NODE_N-1-j,size-i);
    }
  }

  index+=arr[size]-arr[size-1];

  return index;
}

int H_C(int n, int a){
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

void genScore_ref(const int sizepernode,
                  float *D_localscore,
                  const int *D_data,
                  const float *D_LG)
{
  #pragma omp parallel for
  for (int id = 0; id < sizepernode; id++) {
    int node,index;
    bool flag;
    int parent[5]={0};
    int pre[NODE_N]={0};
    int state[5]={0};
    int i,j,parN=0,tmp,t;
    int t1=0,t2=0;
    float ls=0;
    int Nij[STATE_N]={0};

    H_findComb(parent,id,NODE_N-1);

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

      //priors
      t=0;
      while(H_getState(parN,state,t++)){   // for get state
        //printf("test %u\n",id);
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

// slightly modified from AI codes
void computeKernel_ref(
    int sizepernode,
    const float* localscore,
    const bool* parentMask,
    int node,
    int total,
    float &bestScore,
    int bestParents[4])
{
    bestScore = -999999999999999.0f;

    for (int i = 0; i < 4; i++)
        bestParents[i] = 0;

    // pre[1..posN-1] contains candidate parents.
    int pre[NODE_N] = {0};

    int posN = 1;

    for (int i = 0; i < NODE_N; i++)
    {
        if (parentMask[i])
            pre[posN++] = i;
    }

    // Enumerate all tasks.
    for (int task = 0; task < total; task++)
    {
        int parent[5] = {-1};
        int parN = 0;

        H_findComb(parent, task, posN-1);

        // Convert combination indices into actual node ids.
        for (parN = 0; parN < 4; parN++)
        {
            if (parent[parN] < 0)
                break;

            if (pre[parent[parN]] > node)
                parent[parN] = pre[parent[parN]];
            else
                parent[parN] = pre[parent[parN]] + 1;
        }

        // Insert leading 0 expected by D_findindex().
        for (int t = parN; t > 0; --t)
            parent[t] = parent[t - 1];

        parent[0] = 0;

        int index = H_findindex(parent, parN);
        index += sizepernode * node;

        float score = localscore[index];

        if (score > bestScore)
        {
            bestScore = score;

            for (int t = 0; t < 4; t++)
                bestParents[t] = parent[t + 1];
        }
    }
}

// slightly modified from AI codes
float findBestGraph_ref(float* localscore)
{
    float score = 0.0f;

    for (int i = 0; i < NODE_N; ++i)
        for (int j = 0; j < NODE_N; ++j)
            graph[i][j] = 0;

    for (int node = 0; node < NODE_N; ++node)
    {
        // Find best parent set for this node.
        float bestScore;
        int bestParents[4];

        int posN = 0;

        for (int i = 0; i < NODE_N; ++i)
        {
            if (orders[node][i]) ++posN;
        }

        int total = C(posN,4) + C(posN,3) + C(posN,2) + posN + 1;

        computeKernel_ref(
            sizepernode,
            localscore,
            orders[node],
            node,
            total,
            bestScore,
            bestParents);

        // Determine number of parents selected.
        int bestpN = 0;

        for (int i = 0; i < 4; ++i)
        {
            if (bestParents[i] <= 0)
                break;

            ++bestpN;
        }

        // Add edges to graph.
        if (bestScore > -99999999.0f)
        {
            for (int i = 0; i < bestpN; ++i)
            {
                if (bestParents[i] < node)
                    graph[node][bestParents[i] - 1] = 1;
                else
                    graph[node][bestParents[i]] = 1;
            }

            score += bestScore;
        }
    }

    return score;
}
