#pragma once

#include <string>
#include <vector>
#include <ctime>
#include "rewrite.h"

class AIGMan {
public:
    AIGMan(int mainManager = 1);
    ~AIGMan();

    // commands
    int readFile(const char * path);
    void saveFile(const char * path);
    void printTime();
    void printStats();

    void testCommand(int mode);

    // algorithms
    void strash(bool fCPU = false, bool fRecordTime = true);
    void balance(int sortDecId = 1);
    void rewrite(bool fUseZeros = false, bool fGPUDeduplicate = false);
    void refactor(bool fAlgMFFC = false, bool fUseZeros = false, int cutSize = 12);

    // memory helpers
    void allocHost();
    void allocDevice();
    void toHost();
    void toDevice();
    void clearHost();
    void clearDevice();

    // other helpers
    void resetTime();
    void setAigCreated(int aigCreated);
    void setPrevCmdRewrite(int prevCmdRewrite);
    AIGMan * getAuxAig(const std::string & name);
    void addAuxAig(AIGMan * pManAux);

    // debug uses
    void show();
    void showDevice();

    // host data
    int nObjs;   // 1 (const 1) + nPIs + nNodes (actually POs are a subset of nodes)
    int nPIs;
    int nPOs;
    int nNodes;
    int nLevels; // records the delay of the AIG after executing each command
    int * pFanin0, * pFanin1; // saves 2 * id + 0/1
    int * pOuts;              // saves 2 * id + 0/1
    int * pNumFanouts;
    int * pLevel;  // not in use for now

    std::string moduleName; // name of the circuit module
    std::string modulePath; // the external file path
    std::string moduleInfo; // comments and additional info in the last part of the file
    int deviceAllocated; // indicator of whether device data is allocated

    // device data
    int * d_pnObjs;
    int * d_pnPIs;
    int * d_pnPOs;
    int * d_pnNodes;
    int * d_pFanin0, * d_pFanin1;
    int * d_pOuts;
    int * d_pNumFanouts;
    int * d_pLevel;  // not in use for now

    // auxiliary aig managers used in certain algorithms
    // note, auxiliary aigs should be allocated by algorithms but deallocated by this main manager
    std::vector<AIGMan *> vManAux;

private:
    int readFromMemory(char * pContents, int nFileSize);

    void resetRewriteManager();
    void updateFromRewriteManager(int deduplicateMode = 0);

    int mainManager; // indicator of whether this is the main manager of gpuls
    int aigCreated = 0; // indicator of whether the AIG is created or null

    // time recordings
    double prevAlgTime = 0;
    double totalAlgTime = 0;
    double prevFullTime = 0;
    double totalFullTime = 0;

    // rewrite dedicated data
    int prevCmdRewrite = 0;
    rw::CPUSolver rwMan;

};
