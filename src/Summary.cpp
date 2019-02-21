/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <support@xmrig.com>
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


#include <inttypes.h>
#include <stdio.h>
#include <uv.h>


#include "base/net/Pool.h"
#include "common/cpu/Cpu.h"
#include "common/log/Log.h"
#include "core/Config.h"
#include "core/Controller.h"
#include "crypto/Asm.h"
#include "Mem.h"
#include "Summary.h"
#include "version.h"


#ifndef XMRIG_NO_ASM
static const char *coloredAsmNames[] = {
    "\x1B[1;31mnone\x1B[0m",
    "auto",
    "\x1B[1;32mintel\x1B[0m",
    "\x1B[1;32mryzen\x1B[0m",
    "\x1B[1;32mbulldozer\x1B[0m"
};


inline static const char *asmName(xmrig::Assembly assembly, bool colors)
{
    return colors ? coloredAsmNames[assembly] : xmrig::Asm::toString(assembly);
}
#endif


static void print_memory(xmrig::Config *config) {
#   ifdef _WIN32
    if (config->isColors()) {
        Log::i()->text(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") "%s",
                       "HUGE PAGES", Mem::isHugepagesAvailable() ? "\x1B[1;32mavailable" : "\x1B[01;31munavailable");
    }
    else {
        Log::i()->text(" * %-13s%s", "HUGE PAGES", Mem::isHugepagesAvailable() ? "available" : "unavailable");
    }
#   endif
}


static void print_cpu(xmrig::Config *config)
{
    using namespace xmrig;

    if (config->isColors()) {
        Log::i()->text(GREEN_BOLD(" * ") WHITE_BOLD("%-13s%s (%d)") " %sx64 %sAES %sAVX2",
                       "CPU",
                       Cpu::info()->brand(),
                       Cpu::info()->sockets(),
                       Cpu::info()->isX64()   ? "\x1B[1;32m" : "\x1B[1;31m-",
                       Cpu::info()->hasAES()  ? "\x1B[1;32m" : "\x1B[1;31m-",
                       Cpu::info()->hasAVX2() ? "\x1B[1;32m" : "\x1B[1;31m-");
#       ifndef XMRIG_NO_LIBCPUID
        Log::i()->text(GREEN_BOLD(" * ") WHITE_BOLD("%-13s%.1f MB/%.1f MB"), "CPU L2/L3", Cpu::info()->L2() / 1024.0, Cpu::info()->L3() / 1024.0);
#       endif
    }
    else {
        Log::i()->text(" * %-13s%s (%d) %sx64 %sAES %sAVX2",
                       "CPU",
                       Cpu::info()->brand(),
                       Cpu::info()->sockets(),
                       Cpu::info()->isX64()   ? "" : "-",
                       Cpu::info()->hasAES()  ? "" : "-",
                       Cpu::info()->hasAVX2() ? "" : "-");
#       ifndef XMRIG_NO_LIBCPUID
        Log::i()->text(" * %-13s%.1f MB/%.1f MB", "CPU L2/L3", Cpu::info()->L2() / 1024.0, Cpu::info()->L3() / 1024.0);
#       endif
    }
}


static void print_threads(xmrig::Config *config)
{
    if (config->threadsMode() != xmrig::Config::Advanced) {
        char buf[32] = { 0 };
        if (config->affinity() != -1L) {
            snprintf(buf, sizeof buf, ", affinity=0x%" PRIX64, config->affinity());
        }

        Log::i()->text(config->isColors() ? GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("%d") WHITE_BOLD(", %s, av=%d, %sdonate=%d%%") WHITE_BOLD("%s")
                                          : " * %-13s%d, %s, av=%d, %sdonate=%d%%%s",
                       "THREADS",
                       config->threadsCount(),
                       config->algorithm().name(),
                       config->algoVariant(),
                       config->isColors() && config->donateLevel() == 0 ? "\x1B[1;31m" : "",
                       config->donateLevel(),
                       buf);
    }
    else {
        Log::i()->text(config->isColors() ? GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("%d") WHITE_BOLD(", %s, %sdonate=%d%%")
                                          : " * %-13s%d, %s, %sdonate=%d%%",
                       "THREADS",
                       config->threadsCount(),
                       config->algorithm().name(),
                       config->isColors() && config->donateLevel() == 0 ? "\x1B[1;31m" : "",
                       config->donateLevel());
    }

#   ifndef XMRIG_NO_ASM
    if (config->assembly() == xmrig::ASM_AUTO) {
        const xmrig::Assembly assembly = xmrig::Cpu::info()->assembly();

        Log::i()->text(config->isColors() ? GREEN_BOLD(" * ") WHITE_BOLD("%-13sauto:%s")
                                          : " * %-13sauto:%s", "ASSEMBLY", asmName(assembly, config->isColors()));
    }
    else {
        Log::i()->text(config->isColors() ? GREEN_BOLD(" * ") WHITE_BOLD("%-13s%s") : " * %-13s%s", "ASSEMBLY", asmName(config->assembly(), config->isColors()));
    }
#   endif
}


static void print_commands(xmrig::Config *config)
{
    if (config->isColors()) {
        Log::i()->text(GREEN_BOLD(" * ") WHITE_BOLD("COMMANDS     ") MAGENTA_BOLD("h") WHITE_BOLD("ashrate, ")
                                                                     MAGENTA_BOLD("p") WHITE_BOLD("ause, ")
                                                                     MAGENTA_BOLD("r") WHITE_BOLD("esume"));
    }
    else {
        Log::i()->text(" * COMMANDS     'h' hashrate, 'p' pause, 'r' resume");
    }
}


void Summary::print(xmrig::Controller *controller)
{
    controller->config()->printVersions();
    print_memory(controller->config());
    print_cpu(controller->config());
    print_threads(controller->config());
    controller->config()->printPools();
    controller->config()->printAPI();

    print_commands(controller->config());
}



