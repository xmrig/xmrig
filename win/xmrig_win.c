/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
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
 
#include <windows.h>

#include "options.h"
#include "cpu.h"
#include "utils/applog.h"


BOOL WINAPI ConsoleHandler(DWORD dwType)
{
    switch (dwType) {
    case CTRL_C_EVENT:
        applog(LOG_WARNING, "CTRL_C_EVENT received, exiting");
        proper_exit(0);
        break;

    case CTRL_BREAK_EVENT:
        applog(LOG_WARNING, "CTRL_BREAK_EVENT received, exiting");
        proper_exit(0);
        break;

    default:
        return false;
}

    return true;
}


void proper_exit(int reason) {
    if (opt_background) {
        HWND hcon = GetConsoleWindow();
        if (hcon) {
            // unhide parent command line windows
            ShowWindow(hcon, SW_SHOWMINNOACTIVE);
        }
    }

    exit(reason);
}


void os_specific_init()
{
    if (opt_affinity != -1) {
        affine_to_cpu_mask(-1, opt_affinity);
    }

    SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler, TRUE);

    if (opt_background) {
        HWND hcon = GetConsoleWindow();
        if (hcon) {
            // this method also hide parent command line window
            ShowWindow(hcon, SW_HIDE);
        } else {
            HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
            CloseHandle(h);
            FreeConsole();
        }
    }
}
