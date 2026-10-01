#pragma once
#include <vector>
#include <string>
#include "robin_hood.h"

#define isC(x) ((x) & 1)

namespace rw {

struct Cut {
    int sign, leaves[4];
    unsigned truthtable : 16;
    unsigned value : 11;
    unsigned nLeaves : 4;
    bool used;
};

struct TableNode {
    int val, next;
};

struct Library {
    int pPhases[1 << 16], pPerms[1 << 16], pPerms4[2 * 3 * 4][4], pMap[1 << 16];
    int nNodes[222], nSubgr[222], pSubgr[222][20], fanin0[222][60], fanin1[222][60], isC0[222][60], isC1[222][60];
};

inline void swap(int &a, int &b) {
    int tmp = a;
    a = b;
    b = tmp;
}

inline int max(int a, int b) {
    return a > b ? a : b;
}

inline int id(int x) {
    return x >> 1;
}

class GPUSolver {
public:
    GPUSolver() = delete;
    GPUSolver(int n) { Init(n); }
    ~GPUSolver() { Free(); }

    void Init(int n);
    void Free();
    void GetResults(int n, int *CPUbestSubgraph, Cut *CPUcuts);
    void CopyLib(Library CPUlib);
    void EnumerateAndPreEvaluate(int *level, const std::vector<int> &levelCount, 
                                 int n, int *CPUfanin0, int *CPUfanin1, int *CPUref, bool fUseZeros);
    int EnumerateAndPreEvaluateWave(int currIter, int *level, const std::vector<int> &levelCount, 
                                    int n, int *CPUfanin0, int *CPUfanin1, int *CPUref, bool fUseZeros);
    int ReplaceSubgraphs(int n, int *CPUfanin0, int *CPUfanin1, int *CPUphase, int *CPUreplace);

private:
    int *fanin0, *fanin1, *isComplement0, *isComplement1, *nRef, *bestSubgraph;
    int *phase, *replace;
    TableNode *hashTable, *newTable;
    Cut *cuts, *selectedCuts;
    Library *lib;
};

class CPUSolver {
public:
    CPUSolver() {
        AllocMem();
    }

    ~CPUSolver() {
        FreeMem();
    }

    void AllocMem() {
        fanin0 = new int[MAX_NODE_NUM];
        fanin1 = new int[MAX_NODE_NUM];
        ref = new int[MAX_NODE_NUM];
        phase = new int[MAX_NODE_NUM];
        level = new int[MAX_NODE_NUM];
        isDeleted = new int[MAX_NODE_NUM];
        bestOut = new int[MAX_NODE_NUM];
        order = new int[MAX_NODE_NUM];
        output = new int[MAX_NODE_NUM];
        temp0 = new int[MAX_NODE_NUM];
        temp1 = new int[MAX_NODE_NUM];
        bestCut = new Cut[MAX_NODE_NUM];
        visMark = new int[MAX_NODE_NUM];
    }

    void FreeMem() {
        delete [] fanin0;
        delete [] fanin1;
        delete [] ref;
        delete [] phase;
        delete [] level;
        delete [] isDeleted;
        delete [] bestOut;
        delete [] order;
        delete [] output;
        delete [] temp0;
        delete [] temp1;
        delete [] bestCut;
        delete [] visMark;
    }

    void Rewrite(bool fUseZeros = false, bool GPUReplace = true);
    void RewriteWave(bool fUseZeros = false);
    void Reset(int nInputs, int nOutputs, int nTotal, // need to be called if the previous command is not rewrite
               const int * pFanin0, const int * pFanin1, const int * pOuts); 
    void Init(); 
    void Reorder(); 
    void ReadLibrary(); 
    void EvalAndReplace(int id, Cut &cut); 

    int isRedundant(int node) {
        return node > numInputs && (fanin0[node] == fanin1[node] || fanin0[node] == 1);
    }
    int Fanin(int fanin) {
        if(!isRedundant(id(fanin))) return fanin;
        return isC(fanin) ^ Fanin(fanin1[id(fanin)]);
    }
    
    bool IsSmall() { return n <= 100000; }

    int DeReference(int idx) {
        visMark[idx] = visCnt;
        int num = !isRedundant(idx);
        if(--ref[id(fanin0[idx])] == 0)
            num += DeReference(id(fanin0[idx]));
        if(--ref[id(fanin1[idx])] == 0)
            num += DeReference(id(fanin1[idx]));
        return num;
    }

    void Reference(int idx) {
        if(ref[id(fanin0[idx])]++ == 0)
            Reference(id(fanin0[idx]));
        if(ref[id(fanin1[idx])]++ == 0)
            Reference(id(fanin1[idx]));
    }

    int MarkMFFC(int id, Cut &cut) {
        for(int i = 0; i < cut.nLeaves; i++)
            ref[cut.leaves[i]]++;
        int num = DeReference(id);
        Reference(id);
        for(int i = 0; i < cut.nLeaves; i++)
            ref[cut.leaves[i]]--;
        return num;
    }

    int Eval(int cur, int Class, const std::vector<int> &match) {
        if(match[cur] != -1 && visMark[match[cur]] != visCnt) return 0;
        if(created[cur] == visCnt) return 0;
        created[cur] = visCnt;
        return 1 + Eval(lib.fanin0[Class][cur - 4], Class, match) + 
                   Eval(lib.fanin1[Class][cur - 4], Class, match);
    }

    void AddFanout(int parent, int child) {
        ref[child]++;
    }
    void RemoveFanout(int parent, int child) {
        ref[child]--;
    }

    void TableInsert(int node, int in0, int in1) {
        table[HashValue(in0, in1)] = node;
    }
    void TableDelete(int in0, int in1) {
        table.erase(HashValue(in0, in1));
    }

    void Connect(int node, int in0, int in1) {
        if(in0 > in1)
            swap(in0, in1);
        fanin0[node] = in0;
        fanin1[node] = in1;
        AddFanout(node, id(in0));
        AddFanout(node, id(in1));
        level[node] = max(level[id(in0)], level[id(in1)]) + 1;
        phase[node] = (phase[id(in0)] ^ isC(in0)) & (phase[id(in1)] ^ isC(in1));
        TableInsert(node, in0, in1);
    }

    void Disconnect(int node) {
        RemoveFanout(node, id(fanin0[node]));
        RemoveFanout(node, id(fanin1[node]));
        TableDelete(fanin0[node], fanin1[node]);
        fanin0[node] = fanin1[node] = -1;
    }


    void Delete(int node, int flag = 1) {
        if(node <= numInputs)   return;
        int in0 = id(fanin0[node]), in1 = id(fanin1[node]);
        Disconnect(node);
        if(ref[in0] == 0)
            Delete(in0);
        if(ref[in1] == 0)
            Delete(in1);
        isDeleted[node] = flag;
    }

    int FastDelete(int node) {
        if(node <= numInputs)   return 0;
        int in0 = id(fanin0[node]), in1 = id(fanin1[node]);
        ref[in0]--;
        ref[in1]--;
        int cnt = !isRedundant(node);
        if(ref[in0] == 0)
            cnt += FastDelete(in0);
        if(ref[in1] == 0)
            cnt += FastDelete(in1);
        isDeleted[node] = 1;
        return cnt;
    }

    int And(int in0, int in1) {
        int node = ++n;
        ref[node] = isDeleted[node] = 0;
        if(in0 > in1)
            swap(in0, in1);
        Connect(node, in0, in1);
        return node;
    }

    void Build(int cur, int Class, std::vector<int> &match, const std::vector<int> &isC) {
        if(match[cur] != -1) return;
        int in0 = lib.fanin0[Class][cur - 4], in1 = lib.fanin1[Class][cur - 4];
        Build(in0, Class, match, isC);
        Build(in1, Class, match, isC);
        match[cur] = And(match[in0] * 2 + (lib.isC0[Class][cur - 4] ^ isC[in0]), 
                         match[in1] * 2 + (lib.isC1[Class][cur - 4] ^ isC[in1]));
    }

    void ConnectOutput(int node, int outputId) {
        AddFanout(-outputId - 1, id(node));
        output[outputId] = node;
    }

    void DisconnectOutput(int node, int outputId) {
        RemoveFanout(-outputId - 1, node);
    }

    void Replace(int newNode, int oldNode, int isC) {
        ref[newNode]++;
        Delete(oldNode, 0);
        ref[newNode]--;
        if(ref[newNode] || isC) {
            Connect(oldNode, 1, 2 * newNode + isC);
        } else {
            int in0 = fanin0[newNode], in1 = fanin1[newNode];
            Disconnect(newNode);
            Connect(oldNode, in0, in1);
            isDeleted[newNode] = 1;
        }
    }

    unsigned long long HashValue(int in0, int in1) {
        return 2ULL * in0 * MAX_NODE_NUM + in1;
    }

    void BuildTable() {
        table.clear();
        table.reserve(2 * n);
        for(int i = numInputs + 1; i <= n; i++) if(!isDeleted[i])
            TableInsert(i, fanin0[i], fanin1[i]);
    }

    int TableLookup(int in0, int in1) {
        if(in0 > in1)
            swap(in0, in1);
        unsigned long long hashValue = HashValue(in0, in1);
        int ret = -1;
        if(table.count(hashValue))
            ret = table[hashValue];
        return ret;
    }

    void LevelUpdate() {
        for(int i = 1; i <= numInputs; i++)
            level[i] = 0;
        for(int i = numInputs + 1; i <= n; i++)
            level[i] = max(level[id(fanin0[i])], level[id(fanin1[i])]) + 1;
    }

    int TopoSort(int node) {
        if(level[node] != -1) return level[node];
        return level[node] = 1 + max(TopoSort(id(fanin0[node])), TopoSort(id(fanin1[node])));
    }

    void LevelCount() {
        int maxLevel = 0;
        for(int i = 1; i <= n; i++)
            if(!isDeleted[i]) maxLevel = max(maxLevel, level[i]);
        levelCount = std::vector<int> (maxLevel + 1, 0);
        for(int i = 1; i <= n; i++)
            if(!isDeleted[i]) levelCount[level[i]]++;
        for(int i = 1; i <= maxLevel; i++)
            levelCount[i] += levelCount[i - 1];
        
        nLevels = maxLevel;
    }

    int expected = 0;

    std::vector<int> levelCount;

    int n, numInputs, numOutputs, visCnt = 0;
    int nLevels = -1;
    const int MAX_NODE_NUM = 200000000;
    int *fanin0, *fanin1, *ref, *phase, *level, *isDeleted, *bestOut, *order, *output, *temp0,  *temp1, *visMark;
    int created[1000];

    Cut *bestCut;
    robin_hood::unordered_map<unsigned long long, int> table;
    std::string names;
    Library lib;

    GPUSolver *gpuSolver;
};


} // namespace rw