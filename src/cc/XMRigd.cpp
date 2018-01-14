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

#ifdef WIN32
    #define WIN32_LEAN_AND_MEAN  /* avoid including junk */
    #include <windows.h>
    #include <signal.h>
	#include <chrono>
	#include <thread>
#else
#include <sys/wait.h>
#endif

int main(int argc, char **argv) {

    std::string ownPath(argv[0]);

#if defined(_WIN32) || defined(WIN32)
    int pos = ownPath.rfind('\\');
    std::string xmrigMiner("xmrigMiner.exe");
#else
    int pos = ownPath.rfind('/');
    std::string xmrigMiner("xmrigMiner");
#endif

    std::string xmrigMinerPath = ownPath.substr(0, pos+1) + xmrigMiner;

#ifdef WIN32
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
#ifdef WIN32
    } while (status == EINTR);

	if (status == EINVAL) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5000));
	}
#else
    } while (WEXITSTATUS(status) == EINTR);
#endif
}

