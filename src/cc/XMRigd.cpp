/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2017-     BenDr0id    <ben@graef.in>
 *
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <string>
#include <chrono>
#include <thread>

#ifdef WIN32
    #define WIN32_LEAN_AND_MEAN  /* avoid including junk */
    #include <windows.h>
    #include <signal.h>
#else
    #include <sys/wait.h>
    #include <errno.h>
#endif

#ifndef MINER_EXECUTABLE_NAME
  #define MINER_EXECUTABLE_NAME xmrigMiner
#endif
#define VALUE_TO_STRING(x) #x
#define VALUE(x) VALUE_TO_STRING(x)

int main(int argc, char **argv) {

    std::string ownPath(argv[0]);

#if defined(_WIN32) || defined(WIN32)
    int pos = ownPath.rfind('\\');
    std::string xmrigMiner( VALUE(MINER_EXECUTABLE_NAME) ".exe");
#else
    int pos = ownPath.rfind('/');
    std::string xmrigMiner( VALUE(MINER_EXECUTABLE_NAME) );
#endif

    std::string xmrigMinerPath = ownPath.substr(0, pos+1) + xmrigMiner;

#if defined(_WIN32) || defined(WIN32)
    xmrigMinerPath = "\"" + xmrigMinerPath + "\"";
#endif

    for (int i=1; i < argc; i++){
        xmrigMinerPath += " ";
        xmrigMinerPath += argv[i];
    }

    xmrigMinerPath += " --daemonized";

    int status = 0;

    do {
        status = system(xmrigMinerPath.c_str());

        if (WEXITSTATUS(status) == 139) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            printf("Restarting.");
        }
#if defined(_WIN32) || defined(WIN32)
    } while (status == EINTR || status == 139);

    if (WEXITSTATUS(status) == 139) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        printf("Restarting.");
    }

	if (status == EINVAL) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5000));
	}
#else
    } while (WEXITSTATUS(status) == EINTR || WEXITSTATUS(status) == 139); // segfault
#endif
}

