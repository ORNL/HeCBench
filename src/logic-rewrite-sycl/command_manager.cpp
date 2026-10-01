#include <stdexcept>
#include <iostream>
#include <sstream>
#include <string>
#include "CLI11.hpp"
#include "command_manager.h"
#include "string_utils.h"

using strUtil::descWithDefault;

// *************** command handler implementations ***************

int readHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    if (vLiterals.size() < 2)
        return 1;
    aigman.readFile(vLiterals[1].c_str());
    return 0;
}

int writeHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    if (vLiterals.size() < 2)
        return 1;

    if (!strUtil::endsWith(vLiterals[1], ".aig")) {
        printf("write: only support .aig format!\n");
        return 1;
    }
    aigman.saveFile(vLiterals[1].c_str());
    return 0;
}

int timeHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    aigman.printTime();
    return 0;
}

int printStatsHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    aigman.printStats();
    return 0;
}

int balanceHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    int sortDecId = 1;
    for (int i = 1; i < vLiterals.size(); i++) {
        if (vLiterals[i] == "-s") {
            sortDecId = 0;
            printf("** balance without using id as tie break when sorting.\n");
        }
    }
    aigman.balance(sortDecId);
    return 0;
}

int rewriteHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    bool fUseZeros = false, fGPUDeduplicate = true;
    for (int i = 1; i < vLiterals.size(); i++) {
        if (vLiterals[i] == "-z") {
            fUseZeros = true;
        } else if (vLiterals[i] == "-d") {
            fGPUDeduplicate = true;
        }
    }
    aigman.rewrite(fUseZeros, fGPUDeduplicate);
    return 0;
}

int refactorHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    bool fUseZeros = false, fAlgMFFC = false;
    bool fCutSize = false;
    int cutSize = 12;
    for (int i = 1; i < vLiterals.size(); i++) {
        if (vLiterals[i] == "-z") {
            fUseZeros = true;
        } else if (vLiterals[i] == "-m") {
            fAlgMFFC = true;
        } else if (vLiterals[i] == "-K") {
            fCutSize = true;
            continue;
        }

        if (fCutSize) {
            try {
                cutSize = std::stoi(vLiterals[i]);
            } catch(std::invalid_argument const& ex) {
                printf("Wrong cut size format! Use default value 12. \n");
                cutSize = 12;
            }
            fCutSize = false;
        }
    }
    aigman.refactor(fAlgMFFC, fUseZeros, cutSize);
    return 0;
}

int strashHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    bool fCPU = false;
    for (int i = 1; i < vLiterals.size(); i++) {
        if (vLiterals[i] == "-c") {
            fCPU = true;
        }
    }
    aigman.strash(fCPU, true);
    return 0;
}

int resyn2Handler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    // command:
    // b; rw -d; rf -m; st; b; rw -d; rw -z -d; rw -z -d; b -s; rf -m -z; st; 
    // rw -z -d; rw -z -d; b -s

    int cutSize = 12;

    CLI::App parser("Perform resyn2");
    parser.add_option("-K", cutSize, 
                      descWithDefault("maximum cut size used in refactoring", cutSize));
    
    std::string cmd = "";
    for (int i = 0; i < vLiterals.size(); i++) {
        cmd += vLiterals[i];
        if (i != vLiterals.size() - 1)
            cmd += " ";
    }

    try {
        parser.parse(cmd, true);
    } catch (const CLI::CallForHelp &e) {
        std::cout << parser.help();
        return 0;
    } catch (const CLI::ParseError &e) {
        return 1;
    }

    aigman.balance(1);
    aigman.rewrite(false, true);
    aigman.refactor(true, false, cutSize);
    aigman.strash(false, true);
    aigman.balance(1);
    aigman.rewrite(false, true);
    aigman.rewrite(true, true);
    aigman.rewrite(true, true);
    aigman.balance(0);
    aigman.refactor(true, true, cutSize);
    aigman.strash(false, true);
    aigman.rewrite(true, true);
    aigman.rewrite(true, true);
    aigman.balance(0);
    aigman.strash(false, true);

    return 0;
}


// add an register entry here when add a new command
void CmdMan::registerAllCommands() {
    // basic commands
    registerCommand("read", readHandler);
    registerCommand("write", writeHandler);
    registerCommand("time", timeHandler);
    registerCommand("ps", printStatsHandler);
    registerCommand("print_stats", printStatsHandler);

    // main algorithms
    registerCommand("b", balanceHandler);
    registerCommand("balance", balanceHandler);
    registerCommand("rw", rewriteHandler);
    registerCommand("rewrite", rewriteHandler);
    registerCommand("rf", refactorHandler);
    registerCommand("refactor", refactorHandler);
    registerCommand("st", strashHandler);
    registerCommand("strash", strashHandler);
    registerCommand("resyn2", resyn2Handler);

}

void CmdMan::registerCommand(const std::string & cmd, const CommandHandler & cmdHandlerFunc) {
    if (htCmdFuncs.find(cmd) != htCmdFuncs.end()) {
        printf("Command %s already added!\n", cmd.c_str());
        return;
    }

    htCmdFuncs[cmd] = cmdHandlerFunc;
}

void CmdMan::launchCommand(AIGMan & aigman, const std::string & cmd, const std::vector<std::string> & vLiterals) {
    if (vLiterals.size() == 0 || vLiterals[0] != cmd) {
        printf("Wrong argument format of command %s!\n", cmd.c_str());
        return;
    }

    auto cmdRet = htCmdFuncs.find(cmd);
    if (cmdRet == htCmdFuncs.end()) {
        printf("Command %s not registered, ignored.\n", cmd.c_str());
        return;
    }

    auto & cmdHandler = cmdRet->second;
    int ret = cmdHandler(aigman, vLiterals);
    if (ret == 1) {
        printf("Command %s not recognized or in wrong format!\n", cmd.c_str());
        return;
    }
}

void CmdMan::cliMenuAddCommands(AIGMan & aigman, std::unique_ptr<cli::Menu> & cliMenu) {
    for (const auto & it : htCmdFuncs) {
        const auto & cmd = it.first;

        auto cliFunc = [&](std::ostream& out, std::vector<std::string> vTokens) {
            // vTokens provided by cli does not include cmd as the first element, add a new one
            std::vector<std::string> vLiterals;
            vLiterals.push_back(cmd);
            vLiterals.insert(vLiterals.end(), vTokens.begin(), vTokens.end());

            launchCommand(aigman, cmd, vLiterals);
        };

        cliMenu->Insert(cmd, cliFunc, cmd);
    }
}
