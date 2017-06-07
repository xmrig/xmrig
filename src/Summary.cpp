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


#include <uv.h>


#include "Summary.h"
#include "Console.h"
#include "Cpu.h"
#include "Options.h"
#include "version.h"



static void print_versions()
{
    char *buf = static_cast<char*>(malloc(16));

#   ifdef __GNUC__
    snprintf(buf, 16, " gcc/%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   else
    buf[0] = '\0';
#   endif


    if (Options::i()->colors()) {
        Console::i()->text("\x1B[01;32m * \x1B[01;37mVERSIONS:     \x1B[01;36mXMRig/%s\x1B[01;37m libuv/%s%s", APP_VERSION, uv_version_string(), buf);
    } else {
        Console::i()->text(" * VERSIONS:     XMRig/%s libuv/%s%s", APP_VERSION, uv_version_string(), buf);
    }

    free(buf);
}


static void print_cpu()
{
    if (Options::i()->colors()) {
        Console::i()->text("\x1B[01;32m * \x1B[01;37mCPU:          %s (%d) %sx64 %sAES-NI",
                           Cpu::brand(),
                           Cpu::sockets(),
                           Cpu::isX64() ? "\x1B[01;32m" : "\x1B[01;31m-",
                           Cpu::hasAES() ? "\x1B[01;32m" : "\x1B[01;31m-");
#       ifndef XMRIG_NO_LIBCPUID
        Console::i()->text("\x1B[01;32m * \x1B[01;37mCPU L2/L3:    %.1f MB/%.1f MB", Cpu::l2() / 1024.0, Cpu::l3() / 1024.0);
#       endif
    }
    else {
        Console::i()->text(" * CPU:          %s (%d) %sx64 %sAES-NI", Cpu::brand(), Cpu::sockets(), Cpu::isX64() ? "" : "-", Cpu::hasAES() ? "" : "-");
#       ifndef XMRIG_NO_LIBCPUID
        Console::i()->text(" * CPU L2/L3:    %.1f MB/%.1f MB", Cpu::l2() / 1024.0, Cpu::l3() / 1024.0);
#       endif
    }
}


void Summary::print()
{
    print_versions();
    print_cpu();
}



