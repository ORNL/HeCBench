#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <string>
#include <sstream>
#include "common.h"
#include "rewrite.h"
#include "robin_hood.h"
#include "rewrite_library.inc"

using namespace std;
using namespace rw;

#define BLOCK_SIZE 64
#define RATIO  2.3
#define CUT_SET_SIZE 8
#define TABLE_SIZE 8
#define LARGE_BLOCK_SIZE 512

__managed__ int P;
__managed__ int N; // # of graph nodes
__managed__ int GPUexpected = 0;
__managed__ int replaceHasFullCorrsp = 0;

#define ID(i, j) ((i) * CUT_SET_SIZE + (j))
#define BLOCK_NUMBER(n, m) (((n) + (m) - 1) / (m))

double ENUM_TIME = 0, EVAL_TIME = 0, REPLACE_TIME = 0, REORDER_TIME = 0, REDUNDANCY_TIME = 0;
double COPYBACK_TIME = 0, REID_TIME = 0, CHOICE_TIME = 0;

double wall_time = hrClock();
string time(double t = 0) {
    if (t == 0) t = hrClock();
    stringstream t_ss;
    t_ss << "[" << setprecision(3) << setw(6) << fixed << (float) (t / NS_PER_SEC) << "] ";
    return t_ss.str();
}
ostream& print() {
    return cout << time();
}
#define prt print()



__device__ int CutFindValue(Cut *cut, int *nRef) {
    int value = 0, nOnes = 0;
    for(int i = 0; i < cut->nLeaves; i++) {
        value += nRef[cut->leaves[i]];
        nOnes += (nRef[cut->leaves[i]] == 1);
    }
    if(cut->nLeaves < 2) return 1001;
    if(value > 1000)
        value = 1000;
    if(nOnes > 3)
        value = 5 - nOnes;
    return value;
}

__global__ void Inputs(int *nRef, Cut *cuts, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if(idx <= n) {
        Cut* cut = cuts + ID(idx, 0);
        cut->used = 1;
        cut->sign = 1 << (idx & 31);
        cut->truthtable = 0xAAAA;
        cut->nLeaves = 1;
        cut->leaves[0] = idx;
        cut->value = CutFindValue(cut, nRef);
    }
}

__device__ int CountOnes(unsigned uWord) {
    uWord = (uWord & 0x55555555) + ((uWord>>1) & 0x55555555);
    uWord = (uWord & 0x33333333) + ((uWord>>2) & 0x33333333);
    uWord = (uWord & 0x0F0F0F0F) + ((uWord>>4) & 0x0F0F0F0F);
    uWord = (uWord & 0x00FF00FF) + ((uWord>>8) & 0x00FF00FF);
    return  (uWord & 0x0000FFFF) + (uWord>>16);
}

__device__ int FindCut(int idx, Cut *cuts) {
    for(int i = 0; i < CUT_SET_SIZE; i++)
        if(cuts[ID(idx, i)].used == 0) return i;
    int ans = -1;
    for(int i = 0; i < CUT_SET_SIZE; i++)
        if(cuts[ID(idx, i)].nLeaves > 2) {
            if(ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                ans = i;
        }
    if(ans == -1) {
        for(int i = 0; i < CUT_SET_SIZE; i++)
            if(cuts[ID(idx, i)].nLeaves == 2) {
                if(ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                    ans = i;
            }
    }
    if(ans == -1) {
        for(int i = 0; i < CUT_SET_SIZE; i++)
            if(cuts[ID(idx, i)].nLeaves < 2) {
                if(ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                    ans = i;
            }
    }
    cuts[ID(idx, ans)].used = 0;
    return ans;
}

// a.nLeaves >= b.nLeaves
__device__ int MergeCutOrdered(Cut a, Cut b, Cut* cut) {
    int c = 0, i = 0, k = 0;
    for (c = 0; c < 4; c++)
    {
        if (k == b.nLeaves)
        {
            if (i == a.nLeaves)
            {
                cut->nLeaves = c;
                return 1;
            }
            cut->leaves[c] = a.leaves[i++];
            continue;
        }
        if (i == a.nLeaves)
        {
            if (k == b.nLeaves)
            {
                cut->nLeaves = c;
                return 1;
            }
            cut->leaves[c] = b.leaves[k++];
            continue;
        }
        if (a.leaves[i] < b.leaves[k])
        {
            cut->leaves[c] = a.leaves[i++];
            continue;
        }
        if (a.leaves[i] > b.leaves[k])
        {
            cut->leaves[c] = b.leaves[k++];
            continue;
        }
        cut->leaves[c] = a.leaves[i++];
        k++;
    }
    if (i < a.nLeaves || k < b.nLeaves)
        return 0;
    cut->nLeaves = c;
    return 1;
}

__device__ int MergeCut(Cut a, Cut b, Cut *cut) {
    if(a.nLeaves >= b.nLeaves) {
        if(MergeCutOrdered(a, b, cut) == 0) return 0;
    } else {
        if(MergeCutOrdered(b, a, cut) == 0) return 0;
    }
    cut->sign = a.sign | b.sign;
    cut->used = 1;
    return 1;
}

//check if cut a is a subset of cut b
__device__ int CutDominance(Cut a, Cut b) {
    for(int i = 0; i < a.nLeaves; i++) {
        int ok = 0;
        for(int j = 0; j < b.nLeaves; j++)
            if(b.leaves[j] == a.leaves[i]) ok = 1;
        if(!ok) return 0;
    }
    return 1;
}

//check if cut[id] is redundant, and filter out redundant cuts
__device__ int CutFilter(Cut *cut, int id) {
    for(int i = 0; i < CUT_SET_SIZE; i++) if(i != id && cut[i].used) {
        if(cut[i].nLeaves > cut[id].nLeaves) {
            if((cut[i].sign & cut[id].sign) != cut[id].sign) continue;
            if(CutDominance(cut[id], cut[i]))
                cut[i].used = 0;
        } else {
            if((cut[i].sign & cut[id].sign) != cut[i].sign) continue;
            if(CutDominance(cut[i], cut[id])) {
                cut[id].used = 0;
                return 1;
            }
        }
    }
    return 0;
}

__device__ int CutTruthPhase(Cut x, Cut cut) {
    int phase = 0;
    for(int i = 0; i < cut.nLeaves; i++)
        for(int j = 0; j < x.nLeaves; j++)
            if(x.leaves[j] == cut.leaves[i]) phase |= 1 << i;
    return phase;
}

__device__ int CutTruthSwapAdjacentVars(int uTruth, int iVar) {
    if ( iVar == 0 )
        return (uTruth & 0x99999999) | ((uTruth & 0x22222222) << 1) | ((uTruth & 0x44444444) >> 1);
    if ( iVar == 1 )
        return (uTruth & 0xC3C3C3C3) | ((uTruth & 0x0C0C0C0C) << 2) | ((uTruth & 0x30303030) >> 2);
    if ( iVar == 2 )
        return (uTruth & 0xF00FF00F) | ((uTruth & 0x00F000F0) << 4) | ((uTruth & 0x0F000F00) >> 4);
    return 0;
}

__device__ int CutTruthStretch(int truthtable, int nVar, int phase) {
    int Var = nVar - 1;
    for(int i = 3; i >= 0; i--) if(phase >> i & 1) {
        for(int k = Var; k < i; k++)
            truthtable = CutTruthSwapAdjacentVars(truthtable, k);
        Var--;
    }
    return truthtable;
}

__device__ int CutTruthtable(Cut cut, Cut a, Cut b, int aComplement, int bComplement) {
    int tt0 = aComplement ? ~a.truthtable : a.truthtable;
    int tt1 = bComplement ? ~b.truthtable : b.truthtable;
    tt0 = CutTruthStretch(tt0, a.nLeaves, CutTruthPhase(a, cut));
    tt1 = CutTruthStretch(tt1, b.nLeaves, CutTruthPhase(b, cut));
    return tt0 & tt1;
}

__device__ int CutTruthShrink(int uTruth, int nVars, int Phase) {
    int i, k, Var = 0;
    for ( i = 0; i < 4; i++ )
        if ( Phase & (1 << i) )
        {
            for ( k = i-1; k >= Var; k-- )
                uTruth = CutTruthSwapAdjacentVars( uTruth, k );
            Var++;
        }
    return uTruth;
}

__device__ int MinimizeCutSupport(Cut* cut) {
    int masks[4][2] = {
        { 0x5555, 0xAAAA },
        { 0x3333, 0xCCCC },
        { 0x0F0F, 0xF0F0 },
        { 0x00FF, 0xFF00 }
    };
    int phase = 0, truth = cut->truthtable & 0xFFFF, nLeaves = cut->nLeaves;
    for(int i = 0; i < cut->nLeaves; i++)
        if((truth & masks[i][0]) == ((truth & masks[i][1]) >> (1 << i)))
            nLeaves--;
        else
            phase |= 1 << i;
    if(nLeaves == cut->nLeaves) return 0;
    truth = CutTruthShrink(truth, cut->nLeaves, phase);
    cut->truthtable = truth & 0xFFFF;
    cut->sign = 0;
    for(int i = 0, k = 0; i < cut->nLeaves; i++) if(phase >> i & 1) {
        cut->leaves[k++] = cut->leaves[i];
        cut->sign |= 1 << (31 & cut->leaves[i]);
    }
    cut->nLeaves = nLeaves;
    return 1;
}

__global__ void CutEnumerate(int *fanin0, int *fanin1, int* isC0, int *isC1, int *nRef, Cut *cuts, int delta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        idx += delta + 1;
        Cut* cut = cuts + ID(idx, 0);
        cut->used = 1;
        cut->sign = 1 << (idx & 31);
        cut->truthtable = 0xAAAA;
        cut->nLeaves = 1;
        cut->leaves[0] = idx;
        cut->value = CutFindValue(cut, nRef);
        int in0 = fanin0[idx], in1 = fanin1[idx];
        for(int i = 0; i < CUT_SET_SIZE; i++) if(cuts[ID(in0, i)].used == 1)
            for(int j = 0; j < CUT_SET_SIZE; j++) if(cuts[ID(in1, j)].used == 1) {
                Cut a = cuts[ID(in0, i)], b = cuts[ID(in1, j)];
                if(CountOnes(a.sign | b.sign) > 4) continue;
                int cutId = FindCut(idx, cuts);
                if(!MergeCut(a, b, cut + cutId)) continue;
                if(CutFilter(cut, cutId)) continue;
                cut[cutId].truthtable = 0xFFFF & CutTruthtable(cut[cutId], a, b, isC0[idx], isC1[idx]);
                if(MinimizeCutSupport(cut + cutId))
                    CutFilter(cut, cutId);
                cut[cutId].value = CutFindValue(cut + cutId, nRef);
                if(cut[cutId].nLeaves < 2) return;
            }
    }
}

__device__ int Decrease(int id, int *tableSize, int *tableId, int *tableNum, int *nRef, Cut *cut) {
    for(int i = 0; i < cut->nLeaves; i++)
        if(id == cut->leaves[i]) return 1;
    for(int i = 0; i < tableSize[0]; i++)
        if(tableId[i] == id) return --tableNum[i];
    if(tableSize[0] == TABLE_SIZE) {
        //printf("Warning in Decrease\n");
        return 1;
    }
    tableId[tableSize[0]] = id;
    tableNum[tableSize[0]] = nRef[id] - 1;
    tableSize[0]++;
    return nRef[id] - 1;
}


__device__ int IsDeleted(int id, int *tableSize, int *tableId, int *tableNum) {
    for(int i = 0; i < tableSize[0]; i++)
        if(tableId[i] == id) return tableNum[i] == 0;
    return 0;
}


__device__ int CalcMFFC(int cur, Cut* cut, int *fanin0, int *fanin1, int *tableSize, int *tableId, int *tableNum, int *nRef, int root) {
    int ans = 1;
    if(Decrease(fanin0[cur], tableSize, tableId, tableNum, nRef, cut) == 0)
        ans += CalcMFFC(fanin0[cur], cut, fanin0, fanin1, tableSize, tableId, tableNum, nRef, root);
    if(Decrease(fanin1[cur], tableSize, tableId, tableNum, nRef, cut) == 0)
        ans += CalcMFFC(fanin1[cur], cut, fanin0, fanin1, tableSize, tableId, tableNum, nRef, root);
    return ans;
}

__device__ void TableInsert(int in0, int in1, int C0, int C1, TableNode* hashTable, int idx, int offset=0) {
    int index = idx - offset - 1;
    unsigned long key = 0;
    key ^= in0 * 7937;
    key ^= in1 * 2971;
    key ^= C0 * 911;
    key ^= C1 * 353;
    key %= P;
    hashTable[P + index].val = idx;
    while(true) {
        int res = atomicCAS(&hashTable[key].next, -1, P + index);
        if(res == -1)
            break;
        else
            key = res;
    }
}

__device__ int TableLookup(int in0, int in1, int C0, int C1, TableNode* hashTable, int *fanin0, int *fanin1, int *isC0, int *isC1) {
    unsigned long key = 0;
    if(in0 > in1) {
        int temp = in0;
        in0 = in1;
        in1 = temp;
        temp = C0;
        C0 = C1;
        C1 = temp;
    }
    key ^= in0 * 7937;
    key ^= in1 * 2971;
    key ^= C0 * 911;
    key ^= C1 * 353;
    key %= P;
    for(int cur = hashTable[key].next; cur != -1; cur = hashTable[cur].next) {
        int val = hashTable[cur].val;
        if(fanin0[val] == in0 && fanin1[val] == in1 && isC0[val] == C0 && isC1[val] == C1)
            return val;
    }
    return -1;
}

__device__ int Eval(int cur, int *match, int Class, Library *lib, int curId) {
    if(match[cur] > -1) return 0;
    if(match[cur] == curId) return 0;
    match[cur] = curId;
    return 1 + Eval(lib->fanin0[Class][cur - 4], match, Class, lib, curId) + Eval(lib->fanin1[Class][cur - 4], match, Class, lib, curId);
}



__global__ void EvaluateNode(int sz, int *bestout, int *fanin0, int *fanin1, int *isC0, int *isC1, int *nodeLevels, Cut *cuts, Cut* selectedCuts, int *nRef, 
                             Library *lib, TableNode* hashTable, int fUseZeros) {
    if(blockIdx.x * blockDim.x + threadIdx.x >= sz) return;
    int id = 1 + blockIdx.x * blockDim.x + threadIdx.x, reduction = -1, bestLevel = 99999999, bestCut = -1, bestOut;
    int match[54], tableSize, tableId[TABLE_SIZE], tableNum[TABLE_SIZE];
    int matchLevel[54];
    for(int i = 0; i < CUT_SET_SIZE; i++) {
        Cut *cut = cuts + ID(id, i);
        if(cut->used == 0 || cut->nLeaves < 3) continue;
        int nleaves = cut->nLeaves;
        if(nleaves == 3)
            cut->leaves[cut->nLeaves++] = 0;
        tableSize = 0;
        int saved = CalcMFFC(id, cut, fanin0, fanin1, &tableSize, tableId, tableNum, nRef, id);
        int uPhase = lib->pPhases[cut->truthtable];
        int Class = lib->pMap[cut->truthtable];
        int *pPerm = lib->pPerms4[lib->pPerms[cut->truthtable]];
        for(int j = 0; j < 54; j++)
            match[j] = matchLevel[j] = -1;
	      uint64_t isC = 0;
        for(int j = 0; j < 4; j++) {
            match[j] = cut->leaves[pPerm[j]];
            matchLevel[j] = nodeLevels[match[j]];
            if(uPhase >> j & 1)
		            isC |= 1LL << j;
        }
        for(int j = 0; j < lib->nNodes[Class]; j++) {
            int num = j + 4;
            int in0 = lib->fanin0[Class][j], in1 = lib->fanin1[Class][j];
            assert(matchLevel[in0] != -1 && matchLevel[in1] != -1);
            matchLevel[num] = 1 + (matchLevel[in0] > matchLevel[in1] ? matchLevel[in0] : matchLevel[in1]);

            if(match[in0] == -1 || match[in1] == -1 || match[in0] == id || match[in1] == id) continue;
            int nodeId = TableLookup(match[in0], match[in1], (isC >> in0 & 1) ^ lib->isC0[Class][j], (isC >> in1 & 1) ^ lib->isC1[Class][j], hashTable, fanin0, fanin1, isC0, isC1);
            if(nodeId != -1 && !IsDeleted(nodeId, &tableSize, tableId, tableNum)) {
                match[num] = nodeId;
                matchLevel[num] = nodeLevels[nodeId];
            }
        }
        for(int out = 0; out < lib->nSubgr[Class]; out++) {
            int rt = lib->pSubgr[Class][out];
            if(match[rt] == id) continue;
            int nodesAdded = Eval(rt, match, Class, lib, -out - 2);
            int rtLevel = matchLevel[rt];
            assert(rtLevel != -1);
            // if(saved - nodesAdded > reduction || (fUseZeros && saved == nodesAdded && bestCut == -1)) {
            // if(saved - nodesAdded > reduction || (fUseZeros && saved == nodesAdded)) {
            //     reduction = saved - nodesAdded;
            //     bestCut = i;
            //     bestOut = out;
            // }
            if (saved - nodesAdded < 0 || (saved - nodesAdded == 0 && !fUseZeros))
                continue;
            if (saved - nodesAdded < reduction || (saved - nodesAdded == reduction && rtLevel >= bestLevel))
                continue;
            reduction = saved - nodesAdded;
            bestLevel = rtLevel;
            bestCut = i;
            bestOut = out;
        }
        cut->nLeaves = nleaves;
    }
    if(bestCut != -1) {
	// atomicAdd(&GPUexpected, 1);
        selectedCuts[id] = cuts[ID(id, bestCut)];
        selectedCuts[id].used = 1;
        bestout[id] = bestOut;
    } else
        selectedCuts[id].used = 0;
}


__global__ void BuildHashTable(TableNode *hashTable, int sz, int *fanin0, int *fanin1, int *isC0, int *isC1) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < sz) {
        int id = index + 1;
        if(fanin0[id] == 0 && fanin1[id] == 0) return;
        unsigned long key = 0;
        key ^= fanin0[id] * 7937;
        key ^= fanin1[id] * 2971;
        key ^= isC0[id] * 911;
        key ^= isC1[id] * 353;
        key %= P;
        hashTable[P + index].val = id;
        while(true) {
            int res = atomicCAS(&hashTable[key].next, -1, P + index);
            if(res == -1)
                break;
            else
                key = res;
        }
    }

}

__device__ void BuildSubgr(int cur, int Class, Library *lib, int *match, uint64_t isC, int *fanin0, int *fanin1, int *isC0, int *isC1, int *phase, TableNode* newTable, int sz) {
    // only create the node that has no correspondence in original table, or not created by this thread (in this subgraph)
    if (match[cur] != -1) return;
    int in0 = lib->fanin0[Class][cur - 4];
    int in1 = lib->fanin1[Class][cur - 4];
    BuildSubgr(in0, Class, lib, match, isC, fanin0, fanin1, isC0, isC1, phase, newTable, sz);
    BuildSubgr(in1, Class, lib, match, isC, fanin0, fanin1, isC0, isC1, phase, newTable, sz);

    // Build the node
    int node = atomicAdd(&N, 1);
    // assert(node < (int) (sz * RATIO));
    node += 1;
    fanin0[node] = match[in0];
    fanin1[node] = match[in1];
    isC0[node] = lib->isC0[Class][cur - 4] ^ (isC >> in0 & 1);
    isC1[node] = lib->isC1[Class][cur - 4] ^ (isC >> in1 & 1);
    phase[node] = (phase[fanin0[node]] ^ isC0[node]) & (phase[fanin1[node]] ^ isC1[node]);

    if (fanin0[node] > fanin1[node]) {
        int temp = fanin0[node];
        fanin0[node] = fanin1[node];
        fanin1[node] = temp;
        temp = isC0[node];
        isC0[node] = isC1[node];
        isC1[node] = temp;
    }

    // newTable is used for ensuring unique insertion
    TableInsert(fanin0[node], fanin1[node], isC0[node], isC1[node], newTable, node, sz);
    int actual_node = TableLookup(fanin0[node], fanin1[node], isC0[node], isC1[node], newTable, fanin0, fanin1, isC0, isC1);
    if (actual_node != node) { // The node is already created (by other threads, in other subgraphs)
        fanin0[node] = fanin1[node] = -1;
        match[cur] = actual_node;
    } else {
        match[cur] = node;
    }
}

__global__ void ReplaceSubgr(int sz, int *bestout, int *fanin0, int *fanin1, int *isC0, int *isC1, Cut *selectedCuts, Library *lib, TableNode* hashTable, TableNode* newTable, int *phase, int *replace) {
    if(blockIdx.x * blockDim.x + threadIdx.x >= sz) return;
    //if (N >= (int) sz * RATIO) printf("Warning: too many new nodes to handle...\n");
    int id = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    Cut cut = selectedCuts[id];
    if (cut.used == 0 || cut.nLeaves < 3) return;
	//atomicAdd(&CNT, 1);
    if(cut.nLeaves == 3)
        cut.leaves[cut.nLeaves++] = 0;

    int uPhase = lib->pPhases[cut.truthtable];
    int Class = lib->pMap[cut.truthtable];
    int *pPerm = lib->pPerms4[lib->pPerms[cut.truthtable]];
    // match maps the new node id into the id in the original hashtable;
    // if there is no corresponding node in the original table, then match = -1
    int match[54];
	  uint64_t isC = 0;
    for(int j = 0; j < 54; j++) match[j] = -1;
    for(int j = 0; j < 4; j++) {
        match[j] = cut.leaves[pPerm[j]];
        if(uPhase >> j & 1)
        	  isC |= 1LL << j;
    }
    int rt = lib->pSubgr[Class][bestout[id]];
	  uint64_t used = 1LL << rt;
    for(int j = rt; j >= 4; j--) if(used >> j & 1) {
	      used |= 1LL << lib->fanin0[Class][j - 4];
	      used |= 1LL << lib->fanin1[Class][j - 4];
    }
    for(int j = 0; j < lib->nNodes[Class]; j++) if(used >> (j + 4) & 1) {
      	int num = j + 4;
      	int in0 = lib->fanin0[Class][j], in1 = lib->fanin1[Class][j];
      	if(match[in0] == -1 || match[in1] == -1) continue;
      	int nodeId = TableLookup(match[in0], match[in1], (isC >> in0 & 1) ^ lib->isC0[Class][j], (isC >> in1 & 1) ^ lib->isC1[Class][j], hashTable, fanin0, fanin1, isC0, isC1);
      	if(nodeId != -1) match[num] = nodeId;
    }

    // the root of the new graph has correspondence, so the whole new graph exists in the original table
    // if (match[rt] == id) {atomicAdd(&replaceHasFullCorrsp, 1);  return;}
    if (match[rt] == id) return;
    BuildSubgr(rt, Class, lib, match, isC, fanin0, fanin1, isC0, isC1, phase, newTable, sz);
    replace[id] = (match[rt] * 2) + (phase[match[rt]] ^ phase[id]); // the corresponding new node
    // if (id > 100300 && id < 100400)
    //     printf("replace[%d] = %d\n", id, replace[id]);
}

__global__ void DetachAndAttach(int sz, int *fanin0, int *fanin1, int *replace) {
    if(blockIdx.x * blockDim.x + threadIdx.x >= sz) return;
    int id = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (replace[id] == -1) return;
    fanin0[id] = 1;
    fanin1[id] = replace[id];
}

int IsPrime(int n) {
    for(int i = 2; i * i <= n; i++)
        if(n % i == 0) return 0;
    return 1;
}

int GetPrime(int n) {
    while(!IsPrime(n))
        n++;
    return n;
}

void ShowMemory() {
    size_t free_byte ;

    size_t total_byte ;

    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){

        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

        exit(1);

    }

    double free_db = (double)free_byte ;

    double total_db = (double)total_byte ;

    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

__global__ void printCuts(int id, Cut *cuts) {
    for(int i = 0; i < CUT_SET_SIZE; i++) {
        Cut *cut = cuts + ID(id, i);
        printf("cut#%d details: truthtable %d, used%d, nLeaves=%d, leaves=%d %d %d %d\n", i, cut->truthtable, cut->used, cut->nLeaves, cut->leaves[0], cut->leaves[1], cut->leaves[2], cut->leaves[3]);
    }
}


__global__ void Convert(int *fanin, int *isC, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n) {
        isC[id] = fanin[id] & 1;
        fanin[id] >>= 1;
    }
}

__global__ void Revert(int *fanin, int *isC, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id <= n) {
        fanin[id] <<= 1;
        fanin[id] += isC[id];
    }
}

__global__ void print(int n, Cut *selected, int *bestOut) {
    for(int i = 1; i <= n; i++) if(selected[i].used == 1)
        printf("Selected %d: %d %d\n", i, selected[i].truthtable, selected[i].sign);
}




void GPUSolver::Init(int n) {
    P = GetPrime(n + 1);
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    cudaStackSize = limit;
    printf("GPUSolver: setting cudaLimitStackSize = %lu\n", limit * 12);
    cudaDeviceSetLimit(cudaLimitStackSize, limit * 12);
    cudaMalloc(&fanin0, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&fanin1, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&nRef, (n + 1) * sizeof(int));
    cudaMalloc(&isComplement0, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&isComplement1, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&bestSubgraph, (n + 1) * sizeof(int));
    cudaMalloc(&phase, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&replace, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&lib, sizeof(Library));
    ShowMemory();
    cudaMalloc(&newTable, (2 * n + 1 + P) * sizeof(TableNode));
    cudaMalloc(&hashTable, (2 * n + 1 + P) * sizeof(TableNode));
    ShowMemory();
    cudaMalloc(&selectedCuts, (n + 1) * sizeof(Cut));
    ShowMemory();
    cudaMalloc(&cuts, sizeof(Cut) * CUT_SET_SIZE * (n + 1) );
    ShowMemory();
}

void GPUSolver::Free() {
    cudaFree(fanin0);
    cudaFree(fanin1);
    cudaFree(nRef);
    cudaFree(isComplement0);
    cudaFree(isComplement1);
    cudaFree(bestSubgraph);
    cudaFree(phase);
    cudaFree(replace);
    cudaFree(lib);
    cudaFree(newTable);
    cudaFree(hashTable);
    cudaFree(selectedCuts);
    cudaFree(cuts);

    printf("GPUSolver: setting cudaLimitStackSize = %lu\n", cudaStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize, cudaStackSize);
}

void GPUSolver::GetResults(int n, int *CPUbestSubgraph, Cut *CPUcuts) {
    cudaMemcpy(CPUbestSubgraph, bestSubgraph, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(CPUcuts, selectedCuts, (n + 1) * sizeof(Cut), cudaMemcpyDeviceToHost);
}

void GPUSolver::CopyLib(Library CPUlib) {
    cudaMemcpy(lib, &CPUlib, sizeof(Library), cudaMemcpyHostToDevice);
}

void GPUSolver::EnumerateAndPreEvaluate(int *level, const vector<int> &levelCount, int n, int *CPUfanin0, int *CPUfanin1, int *CPUref, bool fUseZeros) {
    int * nodeLevels = phase; // note, phase has not been used yet, so use its memory for now
    cudaMemcpy(nodeLevels, level, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(fanin0, CPUfanin0, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(fanin1, CPUfanin1, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(nRef, CPUref, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(hashTable, -1, sizeof(TableNode) * (2 * n + 1 + P));
    cudaMemset(cuts, 0, CUT_SET_SIZE * (n + 1) * sizeof(Cut));
    auto startTime = hrClock();
    Convert<<<BLOCK_NUMBER(n + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (fanin0, isComplement0, (n + 1));
    Convert<<<BLOCK_NUMBER(n + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (fanin1, isComplement1, (n + 1));
    Inputs<<<BLOCK_NUMBER(levelCount[0], LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (nRef, cuts, levelCount[0]);        
    for(int i = 1; i < levelCount.size(); i++)
        CutEnumerate<<<BLOCK_NUMBER(levelCount[i] - levelCount[i - 1], LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (fanin0, fanin1, isComplement0, isComplement1, nRef, cuts, levelCount[i - 1], levelCount[i] - levelCount[i - 1]);
    cudaDeviceSynchronize();
    ENUM_TIME += hrClock() - startTime;
    startTime = hrClock();
    BuildHashTable<<<BLOCK_NUMBER(n, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (hashTable, n, fanin0, fanin1, isComplement0, isComplement1);           
    EvaluateNode<<<BLOCK_NUMBER(n, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (n, bestSubgraph, fanin0, fanin1, isComplement0, isComplement1, nodeLevels, cuts, selectedCuts, nRef, lib, hashTable, fUseZeros == true);
    gpuErrchk( cudaDeviceSynchronize() );
    std::cerr << cudaGetLastError() << " in EvaluateNode " << std::endl;

    EVAL_TIME += hrClock() - startTime;
} 

int GPUSolver::ReplaceSubgraphs(int n, int *CPUfanin0, int *CPUfanin1, int *CPUphase, int *CPUreplace) {
    prt << "Replacing sub-graphs" << endl;
    cudaMemcpy(phase, CPUphase, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(replace, -1, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMemset(newTable, -1, (2 * n + 1 + P) * sizeof(TableNode));
    N = n;
    ReplaceSubgr<<<BLOCK_NUMBER(n, BLOCK_SIZE), BLOCK_SIZE>>>(n, bestSubgraph, fanin0, fanin1, isComplement0, isComplement1, selectedCuts, lib, hashTable, newTable, phase, replace);
    gpuErrchk( cudaDeviceSynchronize() ); // wait until N
    std::cerr << cudaGetLastError() << " after replace " << std::endl;
printf("N = %d   n = %d   n * RATIO = %d\n", N, n, static_cast<int> (RATIO * n));
    Revert<<<BLOCK_NUMBER(N + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>>(fanin0, isComplement0, N + 1);
    Revert<<<BLOCK_NUMBER(N + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>>(fanin1, isComplement1, N + 1);
    cudaDeviceSynchronize();
    // Detach and attach sequentially
    auto startTime = hrClock();
    int *tempIn0 = (int*)malloc(5 * (n + 1) * sizeof(int));
    int *tempIn1 = (int*)malloc(5 * (n + 1) * sizeof(int));
    cudaMemcpy(tempIn0, fanin0, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tempIn1, fanin1, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(CPUreplace, replace, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    COPYBACK_TIME += hrClock() - startTime;

    startTime = hrClock();
    int nn = n;
    int *mapping = new int[N+1];
    memset(mapping, -1, (N+1)*sizeof(int));
    function<int(int)> rebuild = [&](int rId) {
        if (rId <= nn) return rId;
        if (mapping[rId] != -1) return mapping[rId];
        int in0 = (rebuild(tempIn0[rId] / 2) * 2 + (tempIn0[rId] & 1));
        int in1 = (rebuild(tempIn1[rId] / 2) * 2 + (tempIn1[rId] & 1));
        n++;
        CPUfanin0[n] = in0;
        CPUfanin1[n] = in1;
        mapping[rId] = n;
        return n;
    };
    printf("before rebuild, nn = %d\n", nn);

    // replace maps the old node id to new node id, or, indicate the "choice" node id of the old one
    // rebuild makes the new node ids consecutive
    for (int id = 1; id <= nn; id++) {
            //prt << "id " << id << " replace " << (CPUreplace[id] == -1 ? -1 : CPUreplace[id]/2)<< endl;
        if (CPUreplace[id] == -1) continue;
        else CPUreplace[id] = rebuild(CPUreplace[id] / 2) * 2 + (CPUreplace[id] & 1);
    }

    free(mapping);
    free(tempIn0);
    free(tempIn1);

    REID_TIME += hrClock() - startTime;
    return n;
}


void Message(string s) {
    cerr << fixed << setprecision(5) << "[" << 1.0 * hrClock() / NS_PER_SEC << "] " << s << endl;
}




void CPUSolver::Reset(int nInputs, int nOutputs, int nTotal, 
                      const int * pFanin0, const int * pFanin1, const int * pOuts) {
    // this should be called when the previous command is not rewrite
    FreeMem();
    AllocMem();
    
    int nAnds = nTotal - nInputs;
    numInputs = nInputs;
    numOutputs = nOutputs;
    n = nTotal;

    for (int i = 0; i < nAnds; i++) {
        size_t thisIdx = (size_t)(i + 1 + nInputs);
        fanin0[thisIdx] = invertConstTrueFalse(pFanin0[thisIdx]);
        fanin1[thisIdx] = invertConstTrueFalse(pFanin1[thisIdx]);
    }
    for(int i = 0; i < nOutputs; i++) {
        output[i] = invertConstTrueFalse(pOuts[i]);
    }

    // reset data
    memset(created, 0, sizeof(int) * 1000);
    visCnt = 0;
    expected = 0;

    for(int i = numInputs + 1; i <= n; i++)
        if(id(fanin0[i]) == 0 || id(fanin1[i]) == 0)
            printf("CONSTANT INPUTS FOUND\n");
    Reorder();
}

void CPUSolver::Rewrite(bool fUseZeros, bool GPUReplace) {
    expected = 0;
    gpuSolver = new GPUSolver(n);

    prt << "Rewrite Iteration" << endl;
    ReadLibrary();
    gpuSolver->CopyLib(lib);
    LevelCount();
    gpuSolver->EnumerateAndPreEvaluate(level, levelCount, n, fanin0, fanin1, ref, fUseZeros);
    prt << "Finished GPU enumeration and pre-evaluation" << endl;
    auto startTime = hrClock();
    if (GPUReplace) {
        gpuSolver->GetResults(n, bestOut, bestCut);
        int nn = n;
        int *replace = (int*)malloc((n + 1) * sizeof(int));
        n = gpuSolver->ReplaceSubgraphs(n, fanin0, fanin1, phase, replace);
        printf("after rebuild, n = %d\n", n);
        // printf(" ** pre-eval good cones: %d, replace has full corrspondence: %d\n", GPUexpected, replaceHasFullCorrsp);
        auto choiceStartTime = hrClock();
        for (int i = nn + 1; i <= n; i++) { // initially, new nodes are marked as deleted
            isDeleted[i] = 1;
            ref[i] = 0;
        }

        function<int(int)> install = [&](int idx) {
            if (isDeleted[idx]) {
                isDeleted[idx] = 0;
                int cnt = !isRedundant(idx);
                cnt += install(id(fanin0[idx]));
                cnt += install(id(fanin1[idx]));
                ref[id(fanin0[idx])]++;
                ref[id(fanin1[idx])]++;
                return cnt;
            } else {
                return 0;
            }
        };

        int replacedCount = 0, compromisedCount = 0, revertedCount = 0, preEvalRejectCount = 0, smallCutRejectCount = 0, newIdxRejectCount = 0;
        int replacedPosCount = 0, replacedZeroCount = 0;
        for (int i = numInputs + 1; i <= nn; i++) {
            auto& cut = bestCut[i];
            int newIdx = id(replace[i]); // get the corresponding new choice node id
            // if (cut.used == 0 || cut.nLeaves < 3 || newIdx == -1) {preEvalRejectCount++; continue;}
            if (cut.used == 0) {preEvalRejectCount++; continue;}
            if (cut.nLeaves < 3) {smallCutRejectCount++; continue;}
            if (newIdx == -1) {newIdxRejectCount++; continue;}

            if (newIdx == i || id(fanin1[newIdx]) == i) continue;
            if (cut.nLeaves == 3) cut.leaves[cut.nLeaves++] = 0;
            bool compromised = false;  // if cut is damaged, do not perform replacement
            for (int j = 0; j < cut.nLeaves; j++) {
                if (isDeleted[cut.leaves[j]]) {
                    compromised = true;
                }
            }
            if (compromised) {compromisedCount++; continue;}

            int added = install(newIdx);
            ref[newIdx]++;
            int saved = FastDelete(i);
            ref[newIdx]--;
            if (saved - added > 0 || (saved == added && fUseZeros)) {
                replacedCount++;
                if (saved - added > 0)
                    replacedPosCount++;
                else
                    replacedZeroCount++;

                expected += saved - added;
                isDeleted[i] = 0;
                fanin0[i] = 1;
                fanin1[i] = replace[i];
                ref[0]++;
                ref[newIdx]++;
            } else {
                revertedCount++;
                install(i); // revert
                ref[i]++;
                if(ref[newIdx] == 0) FastDelete(newIdx);
                ref[i]--;
            }
        }
        free(replace);
        CHOICE_TIME += hrClock() - choiceStartTime;
        printf("successfully replaced %d cones (pos %d, zero %d), reverted %d cones, compromised %d cones, pre-eval rejected %d cones, small cut reject %d cones, new idx reject %d cones\n", 
               replacedCount, replacedPosCount, replacedZeroCount, revertedCount, compromisedCount, preEvalRejectCount, smallCutRejectCount, newIdxRejectCount);
    } else {
        gpuSolver->GetResults(n, bestOut, bestCut);
        BuildTable();
        prt << "Finished building table" << endl;
        int nn = n;
        for(int i = numInputs + 1; i <= nn; i++) EvalAndReplace(i, bestCut[i]);
    }
    REPLACE_TIME += hrClock() - startTime;

    printf("after replace, n = %d\n", n);


    prt << "Finished eval and replace" << endl;
    startTime = hrClock();
    for(int i = numInputs + 1; i <= n; i++) if(!isDeleted[i]) {
        if(!isRedundant(i)) {
            fanin0[i] = Fanin(fanin0[i]);
            fanin1[i] = Fanin(fanin1[i]);
        } else
            isDeleted[i] = 1;
    }
    for(int i = 0; i < numOutputs; i++)
        output[i] = Fanin(output[i]);
    REDUNDANCY_TIME += hrClock() - startTime;
    startTime = hrClock();
    Reorder(); // remove deleted nodes
    prt << "Rewrite Iteration Ends" << endl;
    // printf("GPU expected: %d\n", GPUexpected);
    printf("real reduction: %d\n", expected);

    printf("** Total Time breakdown: ENUM %.2lf, EVAL %.2lf, REPLACE %.2lf, REORDER %.2lf, REDUNDANCY %.2lf\n",
           ENUM_TIME / NS_PER_SEC, EVAL_TIME / NS_PER_SEC, 
           REPLACE_TIME / NS_PER_SEC, REORDER_TIME / NS_PER_SEC,
           REDUNDANCY_TIME / NS_PER_SEC);
    printf("** Replace Time breakdown: COPYBACK %.2lf, REID %.2lf, CHOICE %.2lf\n", 
           COPYBACK_TIME / NS_PER_SEC, REID_TIME / NS_PER_SEC, CHOICE_TIME / NS_PER_SEC);
    printf("** CPU sequential time: %.2lf sec\n", (REDUNDANCY_TIME + CHOICE_TIME + (double)(hrClock() - startTime)) / NS_PER_SEC);

    delete gpuSolver;
    printf("after rewrite, n = %d\n", n);
}


int currWaveIndices(int n, int * vBuffer, const std::vector<int> & levelCount, int currIter, 
                    int waveWidth = 1, int waveStride = -1) {
    int nLevels = levelCount.size();
    std::vector<int> waveLevels;
    // level 0 are the PIs, the AND nodes starts at level 1
    for (int i = currIter * waveWidth + 1; i < nLevels; i += waveWidth + waveStride - 1) {
        for (int j = i; j < std::min(i + waveWidth, nLevels); j++)
            waveLevels.push_back(j);
        if (waveStride <= 0)
            break;
    }
    if (waveLevels.size() == 0)
        return -1;

    int nWaveNodes = 0;
    int maxLevel = -1;
    for (int level : waveLevels) {
        nWaveNodes += levelCount[level] - levelCount[level - 1];
        maxLevel = std::max(maxLevel, level);
    }
    assert(nWaveNodes > 0);

    int * vWaveMask = (int *) malloc((n + 1) * sizeof(int));
    memset(vWaveMask, 0, (n + 1) * sizeof(int));
    int curr = 0;
    for (int level : waveLevels) {
        for (int i = levelCount[level - 1]; i < levelCount[level]; i++) {
            vWaveMask[i + 1] = 1; // 1 stands for const 0/1
            curr++;
        }
    }
    assert(curr == nWaveNodes);

    cudaMemcpy(vBuffer, vWaveMask, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(vWaveMask);
    return maxLevel;
}

__global__ void EvaluateNodeWave(int sz, int *bestout, int *fanin0, int *fanin1, int *isC0, int *isC1, int *nodeLevels, Cut *cuts, Cut* selectedCuts, int *nRef, 
                                 Library * lib, TableNode* hashTable, const int * vWaveMask, int fUseZeros) {
    if(blockIdx.x * blockDim.x + threadIdx.x >= sz) return;
    int id = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (!vWaveMask[id]) {
        // node not in this wave
        selectedCuts[id].used = 0;
        return;
    }

    int reduction = -1, bestLevel = 99999999, bestCut = -1, bestOut;
    int match[54], tableSize, tableId[TABLE_SIZE], tableNum[TABLE_SIZE];
    int matchLevel[54];
    for(int i = 0; i < CUT_SET_SIZE; i++) {
        Cut *cut = cuts + ID(id, i);
        if(cut->used == 0 || cut->nLeaves < 3) continue;
        int nleaves = cut->nLeaves;
        if(nleaves == 3)
            cut->leaves[cut->nLeaves++] = 0;
        tableSize = 0;
        int saved = CalcMFFC(id, cut, fanin0, fanin1, &tableSize, tableId, tableNum, nRef, id);
        int uPhase = lib->pPhases[cut->truthtable];
        int Class = lib->pMap[cut->truthtable];
        int *pPerm = lib->pPerms4[lib->pPerms[cut->truthtable]];
        for(int j = 0; j < 54; j++)
            match[j] = matchLevel[j] = -1;
	      uint64_t isC = 0;
        for(int j = 0; j < 4; j++) {
            match[j] = cut->leaves[pPerm[j]];
            matchLevel[j] = nodeLevels[match[j]];
            if(uPhase >> j & 1)
		            isC |= 1LL << j;
        }
        for(int j = 0; j < lib->nNodes[Class]; j++) {
            int num = j + 4;
            int in0 = lib->fanin0[Class][j], in1 = lib->fanin1[Class][j];
            assert(matchLevel[in0] != -1 && matchLevel[in1] != -1);
            matchLevel[num] = 1 + (matchLevel[in0] > matchLevel[in1] ? matchLevel[in0] : matchLevel[in1]);

            if(match[in0] == -1 || match[in1] == -1 || match[in0] == id || match[in1] == id) continue;
            int nodeId = TableLookup(match[in0], match[in1], (isC >> in0 & 1) ^ lib->isC0[Class][j], (isC >> in1 & 1) ^ lib->isC1[Class][j], hashTable, fanin0, fanin1, isC0, isC1);
            if(nodeId != -1 && !IsDeleted(nodeId, &tableSize, tableId, tableNum)) {
                match[num] = nodeId;
                matchLevel[num] = nodeLevels[nodeId];
            }
        }
        for(int out = 0; out < lib->nSubgr[Class]; out++) {
            int rt = lib->pSubgr[Class][out];
            if(match[rt] == id) continue;
            int nodesAdded = Eval(rt, match, Class, lib, -out - 2);
            int rtLevel = matchLevel[rt];
            assert(rtLevel != -1);
            // if(saved - nodesAdded > reduction || (fUseZeros && saved == nodesAdded && bestCut == -1)) {
            // if(saved - nodesAdded > reduction || (fUseZeros && saved == nodesAdded)) {
            //     reduction = saved - nodesAdded;
            //     bestCut = i;
            //     bestOut = out;
            // }
            if (saved - nodesAdded < 0 || (saved - nodesAdded == 0 && !fUseZeros))
                continue;
            if (saved - nodesAdded < reduction || (saved - nodesAdded == reduction && rtLevel >= bestLevel))
                continue;
            reduction = saved - nodesAdded;
            bestLevel = rtLevel;
            bestCut = i;
            bestOut = out;
        }
        cut->nLeaves = nleaves;
    }
    if(bestCut != -1) {
	atomicAdd(&GPUexpected, 1);
        selectedCuts[id] = cuts[ID(id, bestCut)];
        selectedCuts[id].used = 1;
        bestout[id] = bestOut;
    } else
        selectedCuts[id].used = 0;
}

int GPUSolver::EnumerateAndPreEvaluateWave(int currIter, int *level, const std::vector<int> &levelCount, 
                                           int n, int *CPUfanin0, int *CPUfanin1, int *CPUref, bool fUseZeros) {
    int waveWidth = 100, waveStride = -1;
    int * vWaveMask = replace; // note, replace has not been used yet, so use its memory for now
    int maxLevel = currWaveIndices(n, vWaveMask, levelCount, currIter, waveWidth, waveStride);
    if (maxLevel == -1)
        return 0;
    assert(maxLevel < levelCount.size());
    
    int * nodeLevels = phase; // note, phase has not been used yet, so use its memory for now
    cudaMemcpy(nodeLevels, level, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(fanin0, CPUfanin0, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(fanin1, CPUfanin1, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(nRef, CPUref, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(hashTable, -1, sizeof(TableNode) * (2 * n + 1 + P));
    cudaMemset(cuts, 0, CUT_SET_SIZE * (n + 1) * sizeof(Cut));

    auto startTime = hrClock();
    Convert<<<BLOCK_NUMBER(n + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (fanin0, isComplement0, (n + 1));
    Convert<<<BLOCK_NUMBER(n + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (fanin1, isComplement1, (n + 1));
    Inputs<<<BLOCK_NUMBER(levelCount[0], LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (nRef, cuts, levelCount[0]);

    for(int i = 1; i <= maxLevel; i++)
        CutEnumerate<<<BLOCK_NUMBER(levelCount[i] - levelCount[i - 1], LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> 
            (fanin0, fanin1, isComplement0, isComplement1, nRef, cuts, levelCount[i - 1], levelCount[i] - levelCount[i - 1]);
    cudaDeviceSynchronize();
    ENUM_TIME += hrClock() - startTime;
    
    startTime = hrClock();
    BuildHashTable<<<BLOCK_NUMBER(n, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>>(hashTable, n, fanin0, fanin1, isComplement0, isComplement1);
    EvaluateNodeWave<<<BLOCK_NUMBER(n, 768), 768>>>(n, bestSubgraph, fanin0, fanin1, isComplement0, isComplement1, 
                                                    nodeLevels, cuts, selectedCuts, nRef, lib, hashTable, vWaveMask, fUseZeros == true);
    gpuErrchk( cudaDeviceSynchronize() );
    std::cerr << "Error code " << cudaGetLastError() << " in EvaluateNode " << std::endl;
    EVAL_TIME += hrClock() - startTime;
    std::cout << "Finished GPU enumeration and pre-evaluation" << std::endl;
    return 1;
}

void CPUSolver::RewriteWave(bool fUseZeros) {
    ReadLibrary();

    for (int it = 0; ; it++) {
        printf("wave iter %d ...\n", it);
        LevelCount();
        expected = 0;
        gpuSolver = new GPUSolver(n);
        gpuSolver->CopyLib(lib);

        if (!gpuSolver->EnumerateAndPreEvaluateWave(it, level, levelCount, n, fanin0, fanin1, ref, fUseZeros)) {
            printf("end at iter %d\n", it);
            delete gpuSolver;
            break;
        }

        auto startTime = hrClock();
        gpuSolver->GetResults(n, bestOut, bestCut);
        int nn = n;
        int *replace = (int*)malloc((n + 1) * sizeof(int));
        n = gpuSolver->ReplaceSubgraphs(n, fanin0, fanin1, phase, replace);
        printf("after rebuild, n = %d\n", n);
        printf(" ** pre-eval good cones: %d, replace has full corrspondence: %d\n", GPUexpected, replaceHasFullCorrsp);
        auto choiceStartTime = hrClock();
        for (int i = nn + 1; i <= n; i++) { // initially, new nodes are marked as deleted
            isDeleted[i] = 1;
            ref[i] = 0;
        }

        function<int(int)> install = [&](int idx) {
            if (isDeleted[idx]) {
                isDeleted[idx] = 0;
                int cnt = !isRedundant(idx);
                cnt += install(id(fanin0[idx]));
                cnt += install(id(fanin1[idx]));
                ref[id(fanin0[idx])]++;
                ref[id(fanin1[idx])]++;
                return cnt;
            } else {
                return 0;
            }
        };

        int replacedCount = 0, compromisedCount = 0, revertedCount = 0, preEvalRejectCount = 0, smallCutRejectCount = 0, newIdxRejectCount = 0;
        int replacedPosCount = 0, replacedZeroCount = 0;
        for (int i = numInputs + 1; i <= nn; i++) {
            auto& cut = bestCut[i];
            int newIdx = id(replace[i]); // get the corresponding new choice node id
            // if (cut.used == 0 || cut.nLeaves < 3 || newIdx == -1) {preEvalRejectCount++; continue;}
            if (cut.used == 0) {preEvalRejectCount++; continue;}
            if (cut.nLeaves < 3) {smallCutRejectCount++; continue;}
            if (newIdx == -1) {newIdxRejectCount++; continue;}

            if (newIdx == i || id(fanin1[newIdx]) == i) continue;
            if (cut.nLeaves == 3) cut.leaves[cut.nLeaves++] = 0;
            bool compromised = false;  // if cut is damaged, do not perform replacement
            for (int j = 0; j < cut.nLeaves; j++) {
                if (isDeleted[cut.leaves[j]]) {
                    compromised = true;
                }
            }
            if (compromised) {compromisedCount++; continue;}

            int added = install(newIdx);
            ref[newIdx]++;
            int saved = FastDelete(i);
            ref[newIdx]--;
            if (saved - added > 0 || (saved == added && fUseZeros)) {
                replacedCount++;
                if (saved - added > 0)
                    replacedPosCount++;
                else
                    replacedZeroCount++;

                expected += saved - added;
                isDeleted[i] = 0;
                fanin0[i] = 1;
                fanin1[i] = replace[i];
                ref[0]++;
                ref[newIdx]++;
            } else {
                revertedCount++;
                install(i); // revert
                ref[i]++;
                if(ref[newIdx] == 0) FastDelete(newIdx);
                ref[i]--;
            }
        }
        free(replace);
        CHOICE_TIME += hrClock() - choiceStartTime;
        printf("successfully replaced %d cones (pos %d, zero %d), reverted %d cones, compromised %d cones, pre-eval rejected %d cones, small cut reject %d cones, new idx reject %d cones\n", 
               replacedCount, replacedPosCount, replacedZeroCount, revertedCount, compromisedCount, preEvalRejectCount, smallCutRejectCount, newIdxRejectCount);
        
        REPLACE_TIME += hrClock() - startTime;
        printf("after replace, n = %d\n", n);
        std::cout << "Finished eval and replace" << std::endl;

        startTime = hrClock();
        for(int i = numInputs + 1; i <= n; i++) if(!isDeleted[i]) {
            if(!isRedundant(i)) {
                fanin0[i] = Fanin(fanin0[i]);
                fanin1[i] = Fanin(fanin1[i]);
            } else
                isDeleted[i] = 1;
        }
        for(int i = 0; i < numOutputs; i++)
            output[i] = Fanin(output[i]);
        REDUNDANCY_TIME += hrClock() - startTime;
        
        Reorder(); // remove deleted nodes

        std::cout << "Rewrite Iteration Ends" << endl;
        // printf("GPU expected: %d\n", GPUexpected);
        printf("real reduction: %d\n", expected);

        printf("** Total Time breakdown: ENUM %.2lf, EVAL %.2lf, REPLACE %.2lf, REORDER %.2lf, REDUNDANCY %.2lf\n",
            ENUM_TIME / NS_PER_SEC, EVAL_TIME / NS_PER_SEC, 
            REPLACE_TIME / NS_PER_SEC, REORDER_TIME / NS_PER_SEC,
            REDUNDANCY_TIME / NS_PER_SEC);
        printf("** Replace Time breakdown: COPYBACK %.2lf, REID %.2lf, CHOICE %.2lf\n", 
            COPYBACK_TIME / NS_PER_SEC, REID_TIME / NS_PER_SEC, CHOICE_TIME / NS_PER_SEC);

        delete gpuSolver;
        printf("after rewrite iter, n = %d\n", n);
        printf("--------------------------------\n");
    }

    printf("End of rewrite wave, n = %d\n", n);
}

void CPUSolver::Init() {
    for(int i = numInputs + 1; i <= n; i++) {
        if(fanin0[i] > fanin1[i]) swap(fanin0[i], fanin1[i]);
    }
    memset(isDeleted, 0, sizeof(int) * (n + 1));
    memset(ref, 0, sizeof(int) * (n + 1));
    for(int i = numInputs + 1; i <= n; i++) {
        ref[id(fanin0[i])]++;
        ref[id(fanin1[i])]++;
        level[i] = max(level[id(fanin0[i])], level[id(fanin1[i])]) + 1;
        phase[i] = (phase[id(fanin0[i])] ^ isC(fanin0[i])) & (phase[id(fanin1[i])] ^ isC(fanin1[i]));
    }
    for(int i = 0; i < numOutputs; i++)
        ref[id(output[i])]++;
}

void CPUSolver::Reorder() {
    auto startTime = hrClock();

    for(int i = 1; i <= n; i++)
        level[i] = i <= numInputs ? 0 : -1;
    auto startTime2 = hrClock();
    
    // skipping the deleted nodes might cause errors, so do not skip them
    // the redundant nodes can be deleted by strash 
    for(int i = numInputs + 1; i <= n; i++) 
        isDeleted[i] = 0;

    for(int i = numInputs + 1; i <= n; i++) 
        if(!isDeleted[i]) TopoSort(i);
    printf(" *** Topo sort time: %.2lf sec\n", (hrClock() - startTime2) / (double) NS_PER_SEC);
    LevelCount();
    int newN = levelCount.back();
    for(int i = n; i >= 1; i--)
        if(!isDeleted[i]) order[i] = levelCount[level[i]]--;
    for(int i = numInputs + 1; i <= n; i++) {
        temp0[i] = fanin0[i];
        temp1[i] = fanin1[i];
    }
    for(int i = numInputs + 1; i <= n; i++) if(!isDeleted[i]) {
        fanin0[order[i]] = order[id(temp0[i])] * 2 + isC(temp0[i]);
    }
    for(int i = numInputs + 1; i <= n; i++) if(!isDeleted[i]) {
        fanin1[order[i]] = order[id(temp1[i])] * 2 + isC(temp1[i]);
    }
    for(int i = 0; i < numOutputs; i++)
        output[i] = order[id(output[i])] * 2 + isC(output[i]);
    n = newN;
    REORDER_TIME += hrClock() - startTime;
    Init();
}

void CPUSolver::ReadLibrary() {
    int cur = 0;		
    for(int i = 0; i < (1 << 16); i++)
        lib.pPhases[i] = LIBRARY[cur++];
    for(int i = 0; i < (1 << 16); i++)
        lib.pPerms[i] = LIBRARY[cur++];
    for(int i = 0; i < 2 * 3 * 4; i++)
        for(int j = 0; j < 4; j++)
            lib.pPerms4[i][j] = LIBRARY[cur++];
    for(int i = 0; i < (1 << 16); i++)
        lib.pMap[i] = LIBRARY[cur++];
    for(int i = 0; i < 222; i++)
        lib.nNodes[i] = LIBRARY[cur++];
    for(int i = 0; i < 222; i++)
        lib.nSubgr[i] = LIBRARY[cur++];
    for(int Class = 0; Class < 222; Class++) {

        for (int i = 0; i < lib.nNodes[Class]; i++) {
            lib.fanin0[Class][i] = LIBRARY[cur++];
            lib.fanin1[Class][i] = LIBRARY[cur++];
            lib.isC0[Class][i] = LIBRARY[cur++];
            lib.isC1[Class][i] = LIBRARY[cur++];
        }
        for(int j = 0; j < lib.nSubgr[Class]; j++)
            lib.pSubgr[Class][j] = LIBRARY[cur++];
    }
}

void CPUSolver::EvalAndReplace(int id, Cut &cut) {
    if(cut.used == 0 || cut.nLeaves < 3) return;
    if(cut.nLeaves == 3)
        cut.leaves[cut.nLeaves++] = 0;
    for(int i = 0; i < cut.nLeaves; i++)
        if(isDeleted[cut.leaves[i]]) return;
    visCnt++;
    int saved = MarkMFFC(id, cut);
    int uPhase = lib.pPhases[cut.truthtable];
    int Class = lib.pMap[cut.truthtable];
    int *pPerm = lib.pPerms4[lib.pPerms[cut.truthtable]];
    vector<int> match(54, -1), isComplement(54, 0);
    for(int i = 0; i < 4; i++) {
        match[i] = cut.leaves[pPerm[i]];
        isComplement[i] = uPhase >> i & 1;
    }
    int rt = lib.pSubgr[Class][bestOut[id]];
    vector<int> used(lib.nNodes[Class] + 4, 0);
    used[rt] = 1;
    for(int i = rt; i >= 4; i--) if(used[i]) {
        used[lib.fanin0[Class][i - 4]] = 1;
        used[lib.fanin1[Class][i - 4]] = 1;
    }
    for(int i = 0; i < lib.nNodes[Class]; i++) if(used[i + 4]) {
        int num = i + 4;
        int in0 = lib.fanin0[Class][i], in1 = lib.fanin1[Class][i];
        if(match[in0] == -1 || match[in1] == -1) continue;
        int nodeId = TableLookup(match[in0] * 2 + (isComplement[in0] ^ lib.isC0[Class][i]), match[in1] * 2 + (isComplement[in1] ^ lib.isC1[Class][i]));
        if(nodeId != -1)
            match[num] = nodeId;
    }
    int added = Eval(rt, Class, match);
    if(match[rt] == id) return;
    if(saved - added > 0) {
        expected += saved - added;
        Build(rt, Class, match, isComplement);
        Replace(match[rt], id, phase[match[rt]] ^ phase[id]);
    }
}


