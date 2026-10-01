#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm> 
#include <cctype>
#include <locale>
#include "CLI11.hpp"
#include "cli/cli.h"
#include "cli/loopscheduler.h"
#include "cli/clilocalsession.h"
#include "aig_manager.h"
#include "command_manager.h"
#include "string_utils.h"

void launchCmd(CmdMan & cmdman, AIGMan & aigman, std::string & cmd) {
    strUtil::trim(cmd);
    if (cmd.length() == 0)
        return;

    std::istringstream cmdReader(cmd);
    std::string literal;
    char cmdFlag = 0;

    std::vector<std::string> vLiterals;
    while (cmdReader >> literal)
        vLiterals.push_back(literal);
    
    std::string & command = vLiterals[0];

    cmdman.launchCommand(aigman, command, vLiterals);
}

void runInteractive(CmdMan & cmdman, AIGMan & aigman) {
    auto rootMenu = std::make_unique<cli::Menu>("gpuls");

    cmdman.cliMenuAddCommands(aigman, rootMenu);
    
    cli::Cli cli( std::move(rootMenu) );
    cli.ExitAction( [](auto& out){ out << "gpuls terminated.\n"; } );

    cli::LoopScheduler scheduler;
    cli::CliLocalTerminalSession localSession(cli, scheduler, std::cout, 400);
    localSession.ExitAction(
        [&scheduler](auto& out)
        {
            out << "terminating ...\n";
            scheduler.Stop();
        }
    );

    scheduler.Run();
}

int main(int argc, char * argv[]) {
    CLI::App app;

    std::string script = "";
    app.add_option("-c,--script", script, "run the provided commands in script mode");

    CLI11_PARSE(app, argc, argv);

    // create managers
    CmdMan cmdman;
    AIGMan aigman;
    if (script.length() > 0) {
        std::cout << "Running in script mode: \"" << script << "\"." << std::endl;
        std::cout << "============================================" << std::endl;

        std::vector<std::string> vCommands;
        strUtil::split(script, vCommands, ";");

        for (std::string & cmd : vCommands)
            launchCmd(cmdman, aigman, cmd);
    } else {
        // interactive mode
        std::cout << "Starting interactive mode" << std::endl;
        runInteractive(cmdman, aigman);
    }

    return 0;
}
