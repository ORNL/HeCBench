#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include "robin_hood.h"
#include "aig_manager.h"
#include "common.h"
#include "balance.h"

int getFileSize(const char * path);
unsigned aigDecodeBinary(char **ppPos);
void aigEncodeBinary(char *buffer, int &cur, unsigned x);



AIGMan::AIGMan(int mainManager):
    nObjs(0), nPIs(0), nPOs(0), nNodes(0), pFanin0(NULL), pFanin1(NULL), pOuts(NULL), pLevel(NULL), 
    moduleName("module"), modulePath(""), moduleInfo(""), deviceAllocated(0), nLevels(0), 
    mainManager(mainManager)
{
    if (mainManager) {
        size_t dynamicHeapSize;
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)2048 * 1024 * 1024); // 2GB
        cudaDeviceGetLimit(&dynamicHeapSize, cudaLimitMallocHeapSize);
        printf(">> Dynamic heap size changed to: %.2lf MB\n", (double)dynamicHeapSize / 1024.0 / 1024.0);
    }
}

AIGMan::~AIGMan() {
    for (AIGMan * pManAux : vManAux)
        delete pManAux;

    clearDevice();
    clearHost();
    cudaDeviceSynchronize();

    if (mainManager) {
        cudaDeviceReset(); // for cuda memcheck
    }
}

void AIGMan::resetTime() {
    prevAlgTime = totalAlgTime = prevFullTime = totalFullTime = 0;
}
void AIGMan::setAigCreated(int aigCreated) { this->aigCreated = aigCreated; }
void AIGMan::setPrevCmdRewrite(int prevCmdRewrite) {
    this->prevCmdRewrite = prevCmdRewrite;
}

AIGMan * AIGMan::getAuxAig(const std::string & name) {
    for (AIGMan * pManAux : vManAux) {
        if (pManAux->moduleName == name)
            return pManAux;
    }
    return NULL;
}

void AIGMan::addAuxAig(AIGMan * pManAux) {
    vManAux.push_back(pManAux);
}


int AIGMan::readFile(const char * path) {
    if (deviceAllocated) {
        printf("readFile: old data on device is cleared!\n");
        clearDevice();
    }

    FILE * pFile;
    char * pContents;
    int nFileSize;

    nFileSize = getFileSize(path);
    pFile = fopen(path, "rb");
    if (!pFile || !nFileSize)
        return 0;
    pContents = (char *) malloc(sizeof(char) * (size_t)(nFileSize));
    size_t res = fread(pContents, nFileSize, 1, pFile);
    fclose(pFile);

    int ret = readFromMemory(pContents, nFileSize);
    free(pContents);
    aigCreated = ret ? 1 : 0;
    prevCmdRewrite = 0;
    if (ret) {
        moduleName = path;
        modulePath = path;
    }
    
    return ret;
}

int AIGMan::readFromMemory(char * pContents, int nFileSize) {
    int nTotal, nInputs, nOutputs, nLatches, nAnds;
    char *pDrivers, *pSymbols, *pCur;
    unsigned uLit0, uLit1, uLit;
    size_t thisIdx, thisFanin0Idx, thisFanin1Idx;

    // remove current network
    clearHost();
    
    // check if the input file format is correct
    if (strncmp(pContents, "aig", 3) != 0 || pContents[3] != ' ') {
        printf("readFromMemory: wrong input file format.\n");
        return 0;
    }

    // read the parameters (M I L O A) in header
    pCur = pContents;
    while (*pCur != ' ') pCur++;
    pCur++;
    // read the number of objects
    nTotal = atoi(pCur);
    while (*pCur != ' ') pCur++;
    pCur++;
    // read the number of inputs
    nInputs = atoi(pCur);
    while (*pCur != ' ') pCur++;
    pCur++;
    // read the number of latches
    nLatches = atoi(pCur);
    while (*pCur != ' ') pCur++;
    pCur++;
    // read the number of outputs
    nOutputs = atoi(pCur);
    while (*pCur != ' ') pCur++;
    pCur++;
    // read the number of nodes
    nAnds = atoi(pCur);
    while (*pCur != ' ' && *pCur != '\n') pCur++;

    // do not support latches
    assert(nLatches == 0);
    // do not support nBad, nConstr, nJust, nFair in header
    assert(*pCur == '\n');
    pCur++;

    // check the parameters
    if (nTotal != nInputs + nAnds) {
        printf("readFromMemory: the number of objects does not match.\n");
        return 0;
    }

    this->nObjs = nTotal + 1; // 1 for constant-one
    this->nPIs = nInputs;
    this->nPOs = nOutputs;
    this->nNodes = nAnds;

    // allocate memory
    allocHost();
    memset(pFanin0, -1, this->nObjs * sizeof(int));
    memset(pFanin1, -1, this->nObjs * sizeof(int));

    // remember the beginning of latch/PO literals
    pDrivers = pCur;
    // scroll to the beginning of the binary data
    for (int i = 0; i < nOutputs;)
        if (*pCur++ == '\n')
            ++i;
    
    // prepare delay data
    std::vector<int> vDelays(nObjs, -1);
    for (int i = 0; i <= nInputs; ++i)
        vDelays[i] = 0;
    
    // parse AND gates
    int maxDelay = -1;
    for (int i = 0; i < nAnds; ++i) {
        // literal of LHS (2x variable idx)
        thisIdx = (size_t)(i + 1 + nInputs);
        uLit = (thisIdx << 1);
        // literal of RHS0
        uLit1 = uLit - aigDecodeBinary(&pCur);
        // literal of RHS1
        uLit0 = uLit1 - aigDecodeBinary(&pCur);
        assert(uLit0 <= uLit1);

        // 2x variable idx + 0/1 indicating complement status
        // literal 0 indicates const false in aiger file, but we want 0 to represent const true
        pFanin0[thisIdx] = invertConstTrueFalse(uLit0);
        pFanin1[thisIdx] = invertConstTrueFalse(uLit1);

        // update num fanouts
        thisFanin0Idx = (size_t)(uLit0 >> 1);
        thisFanin1Idx = (size_t)(uLit1 >> 1);
        ++pNumFanouts[thisFanin0Idx];
        ++pNumFanouts[thisFanin1Idx];

        // update delay
        assert(vDelays[thisFanin0Idx] != -1 && vDelays[thisFanin1Idx] != -1);
        vDelays[thisIdx] = 1 + std::max(vDelays[thisFanin0Idx], vDelays[thisFanin1Idx]);
        maxDelay = std::max(maxDelay, vDelays[thisIdx]);
    }
    this->nLevels = maxDelay;

    // remember the place where symbols begin
    pSymbols = pCur;

    // read the PO driver literals
    pCur = pDrivers;
    for (int i = 0; i < nOutputs; ++i) {
        // 2x variable idx + 0/1 indicating complement status
        pOuts[i] = invertConstTrueFalse(atoi(pCur));
        ++pNumFanouts[(size_t)(pOuts[i] >> 1)];
        while (*pCur++ != '\n');
    }

    // skipping symbols and comments
    pCur = pSymbols;

    return 1;
}

void AIGMan::saveFile(const char * path) {
    if (!aigCreated) {
        printf("saveFile: AIG is null! \n");
        return;
    }

    if (deviceAllocated) {
        toHost();
        clearDevice();
    }

    FILE * file = fopen(path, "wb");
    fprintf(file, "aig %d %d 0 %d %d\n", this->nObjs - 1, this->nPIs, this->nPOs, this->nNodes);
    for(int i = 0; i < this->nPOs; i++) {
        unsigned lit = this->pOuts[i];
        fprintf(file, "%d\n", invertConstTrueFalse(lit));
    }

    char * buffer = new char[this->nNodes * 30];
    int cur = 0;
    for(int i = this->nPIs + 1; i < this->nObjs; i++) {
        int lit0 = invertConstTrueFalse(this->pFanin0[i]);
        int lit1 = invertConstTrueFalse(this->pFanin1[i]);
        assert(2 * i - lit1 >= 0);
        assert(lit1 - lit0 >= 0);
        aigEncodeBinary(buffer, cur, 2 * i - lit1);
        aigEncodeBinary(buffer, cur, lit1 - lit0);
    }
    fwrite(buffer, sizeof(char), cur * sizeof(char), file);
    for(auto e : this->moduleInfo)
        putc(e, file);
    fprintf(file, "c\n");
    fclose(file);
    delete [] buffer;

    printf("Output AIG file saved at path: %s\n", path);
}

void AIGMan::printTime() {
    printf("{time} prev cmd: alg %.2lf s, full %.2lf s; total: alg %.2lf s, full %.2lf s.\n",
           (double) prevAlgTime / NS_PER_SEC, (double) prevFullTime / NS_PER_SEC,
           (double) totalAlgTime / NS_PER_SEC, (double) totalFullTime / NS_PER_SEC);
}


void AIGMan::allocHost() {
    pFanin0 = (int *) malloc(this->nObjs * sizeof(int));
    pFanin1 = (int *) malloc(this->nObjs * sizeof(int));
    pOuts = (int *) malloc(this->nPOs * sizeof(int));
    pNumFanouts = (int *) calloc(this->nObjs, sizeof(int));
}

void AIGMan::allocDevice() {
    cudaMalloc(&d_pnObjs, sizeof(int));
    cudaMalloc(&d_pnPIs, sizeof(int));
    cudaMalloc(&d_pnPOs, sizeof(int));
    cudaMalloc(&d_pnNodes, sizeof(int));
    cudaMalloc(&d_pFanin0, this->nObjs * sizeof(int));
    cudaMalloc(&d_pFanin1, this->nObjs * sizeof(int));
    cudaMalloc(&d_pOuts, this->nPOs * sizeof(int));
    cudaMalloc(&d_pNumFanouts, this->nObjs * sizeof(int));
}

void AIGMan::toDevice() {
    clearDevice();

    allocDevice();
    cudaMemset(d_pNumFanouts, 0, this->nObjs * sizeof(int));

    cudaMemcpy(d_pnObjs, &nObjs, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pnPIs, &nPIs, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pnPOs, &nPOs, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pnNodes, &nNodes, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pFanin0, pFanin0, this->nObjs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pFanin1, pFanin1, this->nObjs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pOuts, pOuts, this->nPOs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pNumFanouts, pNumFanouts, this->nObjs * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    deviceAllocated = 1;
}

void AIGMan::toHost() {
    if (!deviceAllocated) {
        printf("toHost: device data not allocated!\n");
        return;
    }
    clearHost();

    cudaMemcpy(&nObjs, d_pnObjs, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&nPIs, d_pnPIs, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&nPOs, d_pnPOs, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&nNodes, d_pnNodes, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    allocHost();

    cudaMemcpy(pFanin0, d_pFanin0, this->nObjs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pFanin1, d_pFanin1, this->nObjs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pNumFanouts, d_pNumFanouts, this->nObjs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pOuts, d_pOuts, this->nPOs * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(pLevel, d_pLevel, this->nObjs * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

void AIGMan::clearHost() {
    if (nObjs > 0) {
        free(pFanin0);
        free(pFanin1);
        free(pOuts);
        free(pNumFanouts);

        nObjs = nPIs = nPOs = nNodes = 0;
    }
}

void AIGMan::clearDevice() {
    if (deviceAllocated) {
        cudaFree(d_pnObjs);
        cudaFree(d_pnPIs);
        cudaFree(d_pnPOs);
        cudaFree(d_pnNodes);

        cudaFree(d_pFanin0);
        cudaFree(d_pFanin1);
        cudaFree(d_pOuts);
        cudaFree(d_pNumFanouts);

        deviceAllocated = 0;
    }
}

void AIGMan::show() {
    std::unordered_set<int> fanoutZero;

    printf("-------Original AIG-------\n");
    printf("id\tfanin0\tfanin1\tnumFanouts\n");
    for (int i = 0; i < nObjs; i++) {
        printf("%d\t", i);
        if (pFanin0[i] != -1)
            printf("%s%d\t", AigNodeIsComplement(pFanin0[i]) ? "!" : "", AigNodeID(pFanin0[i]));
        else
            printf("\t");
        if (pFanin1[i] != -1)
            printf("%s%d\t", AigNodeIsComplement(pFanin1[i]) ? "!" : "", AigNodeID(pFanin1[i]));
        else
            printf("\t");
        printf("%d", pNumFanouts[i]);
        printf("\n");
        if (pNumFanouts[i] == 0 and i > nPIs)
            fanoutZero.insert(i);
        else if (i > nPIs) {
            assert(AigNodeID(pFanin0[i]) < i);
            assert(AigNodeID(pFanin1[i]) < i);
        }
    }
    for (int i = 0; i < nPOs; i++) {
        printf("%d\t", i + nObjs);
        printf("%s%d\n", AigNodeIsComplement(pOuts[i]) ? "!" : "", AigNodeID(pOuts[i]));
    }

    if (!fanoutZero.empty()) {
        for (auto e : fanoutZero)
            printf("***** fanout=0 node at id %d\n", e);
    }
}

void AIGMan::resetRewriteManager() {
    rwMan.Reset(nPIs, nPOs, nObjs - 1, pFanin0, pFanin1, pOuts);
}


__global__ void processRwmanFanins(int * pFanin0, int * pFanin1, int * pNumFanouts, 
                                   int nPIs, int nNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nNodes) {
        int nodeId = idx + nPIs + 1;
        
        // invert const 0/1
        if (pFanin0[nodeId] < 2)
            pFanin0[nodeId] = 1 - pFanin0[nodeId];
        if (pFanin1[nodeId] < 2)
            pFanin1[nodeId] = 1 - pFanin1[nodeId];
        
        atomicAdd(&pNumFanouts[dUtils::AigNodeID(pFanin0[nodeId])], 1);
        atomicAdd(&pNumFanouts[dUtils::AigNodeID(pFanin1[nodeId])], 1);
    }
}

__global__ void processRwmanOuts(int * pOuts, int * pNumFanouts, int nPOs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPOs) {
        if (pOuts[idx] < 2)
            pOuts[idx] = 1 - pOuts[idx];
        
        atomicAdd(&pNumFanouts[dUtils::AigNodeID(pOuts[idx])], 1);
    }
}

void AIGMan::updateFromRewriteManager(int deduplicateMode) {
    // deduplicateMode:
    // 1: CPU deduplicate, 2: GPU deduplicate, others: no deduplicate
    if (!prevCmdRewrite) {
        printf("updateFromRewriteManager: previous command is not rewrite. Do not update data from reMan.\n");
        return;
    }

    clearDevice(); 
    clearHost();

    // load data
    nPIs = rwMan.numInputs;
    nPOs = rwMan.numOutputs;
    nObjs = rwMan.n + 1;
    nNodes = rwMan.n - rwMan.numInputs;
    nLevels = rwMan.nLevels;
    
    allocHost();
    memset(pFanin0, -1, this->nObjs * sizeof(int));
    memset(pFanin1, -1, this->nObjs * sizeof(int));

    if (deduplicateMode == 1) {
        printf(" before deduplicate, nNodes = %d\n", nNodes);
        auto deduplicateStartTime = hrClock();
        // rewrite might produce duplicate AND nodes, needs another strashing
        int lit0, lit1, id0, id1, idCounter, temp;
        uint64 key;

        robin_hood::unordered_map<uint64, int> strashTable;
        std::vector<int> oldToNew(nObjs, -1);
        strashTable.reserve(nNodes);

        std::function<uint64(int, int)> formKey = [](int lit0, int lit1){
            assert(lit0 <= lit1);
            uint32 uLit0 = (uint32)lit0;
            uint32 uLit1 = (uint32)lit1;
            return ((uint64)uLit0) << 32 | uLit1;
        };

        // initialize oldToNew for PIs
        for (int i = 0; i <= nPIs; i++)
            oldToNew[i] = i;

        idCounter = 1 + nPIs;
        for (int i = 0; i < nNodes; i++) {
            int thisIdx = i + 1 + nPIs;
            lit0 = invertConstTrueFalse(rwMan.fanin0[thisIdx]);
            lit1 = invertConstTrueFalse(rwMan.fanin1[thisIdx]);
            id0 = AigNodeID(lit0), id1 = AigNodeID(lit1);
            
            // map to new id
            lit0 = dUtils::AigNodeLitCond(oldToNew[id0], dUtils::AigNodeIsComplement(lit0));
            lit1 = dUtils::AigNodeLitCond(oldToNew[id1], dUtils::AigNodeIsComplement(lit1));
            if (lit0 > lit1) {
                temp = lit0, lit0 = lit1, lit1 = temp;
            }

            key = formKey(lit0, lit1);

            // strashing
            auto strashRet = strashTable.find(key);
            if (strashRet == strashTable.end()) {
                // new node, insert
                strashTable[key] = idCounter;
                oldToNew[thisIdx] = idCounter;
                // save results
                pFanin0[idCounter] = lit0;
                pFanin1[idCounter] = lit1;
                ++pNumFanouts[AigNodeID(lit0)];
                ++pNumFanouts[AigNodeID(lit1)];
                ++idCounter;
            } else {
                // already exists, retrieve
                oldToNew[thisIdx] = strashRet->second;
            }
        }
        nObjs = idCounter;
        nNodes = nObjs - nPIs - 1;
        printf(" after deduplicate, nNodes = %d, elapsed time = %.2lf sec\n", nNodes, 
               (hrClock() - deduplicateStartTime) / (double) NS_PER_SEC);

        for (int i = 0; i < nPOs; i++) {
            lit0 = invertConstTrueFalse(rwMan.output[i]);
            id0 = AigNodeID(lit0);
            lit0 = dUtils::AigNodeLitCond(oldToNew[id0], dUtils::AigNodeIsComplement(lit0));

            pOuts[i] = lit0;
            ++pNumFanouts[AigNodeID(lit0)];
        }

    } else if (deduplicateMode == 2) {
        // GPU deduplicate
        allocDevice();
        deviceAllocated = 1;

        cudaMemcpy(d_pnObjs, &nObjs, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pnPIs, &nPIs, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pnPOs, &nPOs, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pnNodes, &nNodes, sizeof(int), cudaMemcpyHostToDevice);

        // first, copy from rwMan to GPU memory
        cudaMemcpy(d_pFanin0, rwMan.fanin0, nObjs * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pFanin1, rwMan.fanin1, nObjs * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pOuts, rwMan.output, nPOs * sizeof(int), cudaMemcpyHostToDevice);

        // then, invert const 0/1 and compute number of fanouts
        cudaMemset(d_pNumFanouts, 0, this->nObjs * sizeof(int));
        processRwmanFanins<<<NUM_BLOCKS(nNodes, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            d_pFanin0, d_pFanin1, d_pNumFanouts, nPIs, nNodes);
        processRwmanOuts<<<NUM_BLOCKS(nPOs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(d_pOuts, d_pNumFanouts, nPOs);

        // call GPU strash
        strash(false, false);

    } else {
        for (int i = 0; i < nNodes; i++) {
            // literal of LHS (2x variable idx)
            size_t thisIdx = (size_t)(i + 1 + nPIs);
            pFanin0[thisIdx] = invertConstTrueFalse(rwMan.fanin0[thisIdx]);
            pFanin1[thisIdx] = invertConstTrueFalse(rwMan.fanin1[thisIdx]);

            ++pNumFanouts[(size_t)(pFanin0[thisIdx] >> 1)];
            ++pNumFanouts[(size_t)(pFanin1[thisIdx] >> 1)];
        }

        for (int i = 0; i < nPOs; ++i) {
            // 2x variable idx + 0/1 indicating complement status
            pOuts[i] = invertConstTrueFalse(rwMan.output[i]);
            ++pNumFanouts[(size_t)(pOuts[i] >> 1)];
        }
    }
}

__global__ void showDeviceKernel(int * d_pnObjs, int * d_pnPIs, int * d_pnPOs, int * d_pnNodes, 
                                 int * d_pFanin0, int * d_pFanin1, int * d_pOuts, 
                                 int * d_pNumFanouts, int * d_pLevel) {
    printf("-------Original AIG Device-------\n");
    printf("id\tfanin0\tfanin1\tnumFanouts\n");
    for (int i = 0; i < *d_pnObjs; i++) {
        printf("%d\t", i);
        if (d_pFanin0[i] != -1)
            printf("%s%d\t", dUtils::AigNodeIsComplement(d_pFanin0[i]) ? "!" : "", dUtils::AigNodeID(d_pFanin0[i]));
        else
            printf("\t");
        if (d_pFanin1[i] != -1)
            printf("%s%d\t", dUtils::AigNodeIsComplement(d_pFanin1[i]) ? "!" : "", dUtils::AigNodeID(d_pFanin1[i]));
        else
            printf("\t");
        printf("%d", d_pNumFanouts[i]);
        printf("\n");
    }
    for (int i = 0; i < *d_pnPOs; i++) {
        printf("%d\t", i + *d_pnObjs);
        printf("%s%d\n", dUtils::AigNodeIsComplement(d_pOuts[i]) ? "!" : "", dUtils::AigNodeID(d_pOuts[i]));
    }
    printf("nObjs: %d, nPIs: %d, nPOs:%d, nNodes: %d\n", *d_pnObjs, *d_pnPIs, *d_pnPOs, *d_pnNodes);
}

void AIGMan::showDevice() {
    showDeviceKernel<<<1, 1>>>(
        d_pnObjs, d_pnPIs, d_pnPOs, d_pnNodes, 
        d_pFanin0, d_pFanin1, d_pOuts, 
        d_pNumFanouts, d_pLevel
    );
    cudaDeviceSynchronize();
}

__global__ void printStatsKernel(const int * pnPIs, const int * pnPOs, const int * pnNodes) {
    printf("AIG stats: i/o = %d/%d and = %d", *pnPIs, *pnPOs, *pnNodes);
}

void AIGMan::printStats() {
    if (deviceAllocated) {
        printStatsKernel<<<1, 1>>>(d_pnPIs, d_pnPOs, d_pnNodes);
        cudaDeviceSynchronize();
        printf(" level = %d\n", nLevels);
    } else {
        printf("AIG stats: i/o = %d/%d and = %d level = %d\n", nPIs, nPOs, nNodes, nLevels);
    }
}

/* -------------- Algorithm Main Entrance -------------- */

void AIGMan::rewrite(bool fUseZeros, bool fGPUDeduplicate) {
    if (fUseZeros) {
        printf("rewrite: use zeros activated!\n");
    }
        

    if (!aigCreated) {
        printf("rewrite: AIG is null! \n");
        return;
    }

double startFullTime = hrClock();
    
    // for the main aig manager, copy data to from gpu to host and free gpu data
    if (deviceAllocated) {
        toHost();
        clearDevice();
    }

    if (!prevCmdRewrite)
        resetRewriteManager();
    
double startAlgTime = hrClock();
    rwMan.Rewrite(fUseZeros, true);
prevAlgTime = hrClock() - startAlgTime;
totalAlgTime += prevAlgTime;
    
    prevCmdRewrite = 1;
    updateFromRewriteManager(fGPUDeduplicate ? 2 : 1);

    assert(fGPUDeduplicate ? deviceAllocated : !deviceAllocated);
    printf("rewrite: after rewrite, nNodes = %d\n", nNodes);

prevFullTime = hrClock() - startFullTime;
totalFullTime += prevFullTime;
printf("rewrite: alg time %.2lf, full time %.2lf\n", 
       (double)prevAlgTime / NS_PER_SEC, (double)prevFullTime / NS_PER_SEC);
}

__global__ void updateDeviceStats(const int nEntries, const int nPIs, int * pnNodes, int * pnObjs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        *pnNodes = nEntries;
        *pnObjs = nEntries + nPIs + 1;
    }
}

void AIGMan::balance(int sortDecId) {
    if (!aigCreated) {
        printf("balance: AIG is null! \n");
        return;
    }

double startFullTime = hrClock();

    int * vFanin0, * vFanin1, * vNumFanouts, * vPOs;
    int nEntries;

    // copy data to device
    if (!deviceAllocated)
        toDevice();

double startAlgTime = hrClock();
    std::tie(vFanin0, vFanin1, vNumFanouts, vPOs, nEntries) = balancePerformV2(
        nObjs, nPIs, nPOs, nNodes, 
        d_pFanin0, d_pFanin1, d_pOuts, d_pNumFanouts, sortDecId
    );
prevAlgTime = hrClock() - startAlgTime;
totalAlgTime += prevAlgTime;

    // substitute data structures in AIGMan with balanced results
    updateDeviceStats<<<1, 1>>>(nEntries, nPIs, d_pnNodes, d_pnObjs);
    cudaDeviceSynchronize();

    nNodes = nEntries;
    nObjs = nEntries + nPIs + 1;

    cudaFree(d_pFanin0);
    cudaFree(d_pFanin1);
    cudaFree(d_pOuts);
    cudaFree(d_pNumFanouts);
    d_pFanin0 = vFanin0;
    d_pFanin1 = vFanin1;
    d_pOuts = vPOs;
    d_pNumFanouts = vNumFanouts;

    assert(deviceAllocated);

    nLevels = -1; // the levels of the AIG is not computed in balancing!

    prevCmdRewrite = 0;
prevFullTime = hrClock() - startFullTime;
totalFullTime += prevFullTime;
printf("balance: alg time %.2lf, full time %.2lf\n", 
       (double)prevAlgTime / NS_PER_SEC, (double)prevFullTime / NS_PER_SEC);
}

/* -------------- IO Utils -------------- */

int getFileSize(const char * path) {
    FILE * pFile;
    int nFileSize;
    pFile = fopen(path, "r");
    if (pFile == NULL) {
        printf( "getFileSize: The file is unavailable (absent or open).\n" );
        return 0;
    }
    fseek(pFile, 0, SEEK_END);
    nFileSize = ftell(pFile);
    fclose(pFile);
    return nFileSize;
}

unsigned aigDecodeBinary(char **ppPos) {
    unsigned x = 0, i = 0;
    unsigned char ch;

    while ((ch = *(*ppPos)++) & 0x80)
        x |= (ch & 0x7f) << (7 * i++);

    return x | (ch << (7 * i));
}

void aigEncodeBinary(char *buffer, int &cur, unsigned x) {
    unsigned char ch;
    while(x & ~0x7f) {
        ch = (x & 0x7f) | 0x80;
        buffer[cur++] = ch;
        x >>= 7;
    }
    ch = x;
    buffer[cur++] = ch;
}
