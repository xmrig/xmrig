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


#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Pool.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "crypto/common/Assembly.h"
#include "crypto/common/VirtualMemory.h"
#include "Summary.h"
#include "version.h"


#ifdef XMRIG_FEATURE_ASM
static const char *coloredAsmNames[] = {
    RED_BOLD("none"),
    "auto",
    GREEN_BOLD("intel"),
    GREEN_BOLD("ryzen"),
    GREEN_BOLD("bulldozer")
};


inline static const char *asmName(xmrig::Assembly::Id assembly)
{
    return coloredAsmNames[assembly];
}
#endif


static void print_memory(xmrig::Config *) {
#   ifdef _WIN32
    xmrig::Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") "%s",
                      "HUGE PAGES", xmrig::VirtualMemory::isHugepagesAvailable() ? GREEN_BOLD("available") : RED_BOLD("unavailable"));
#   endif
}


static void print_cpu(xmrig::Config *)
{
    using namespace xmrig;

    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s%s (%d)") " %sx64 %sAES %sAVX2",
               "CPU",
               Cpu::info()->brand(),
               Cpu::info()->sockets(),
               Cpu::info()->isX64()   ? GREEN_BOLD_S : RED_BOLD_S "-",
               Cpu::info()->hasAES()  ? GREEN_BOLD_S : RED_BOLD_S "-",
               Cpu::info()->hasAVX2() ? GREEN_BOLD_S : RED_BOLD_S "-"
               );
#   ifdef XMRIG_FEATURE_LIBCPUID
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s%.1f MB/%.1f MB"), "CPU L2/L3", Cpu::info()->L2() / 1024.0, Cpu::info()->L3() / 1024.0);
#   endif
}


static void print_threads(xmrig::Config *config)
{
    xmrig::Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") WHITE_BOLD("%s%d%%"),
                      "DONATE",
                      config->pools().donateLevel() == 0 ? RED_BOLD_S : "",
                      config->pools().donateLevel()
                      );

#   ifdef XMRIG_FEATURE_ASM
    if (config->cpu().assembly() == xmrig::Assembly::AUTO) {
        const xmrig::Assembly assembly = xmrig::Cpu::info()->assembly();

        xmrig::Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13sauto:%s"), "ASSEMBLY", asmName(assembly));
    }
    else {
        xmrig::Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s%s"), "ASSEMBLY", asmName(config->cpu().assembly()));
    }
#   endif
}


static void print_commands(xmrig::Config *)
{
    if (xmrig::Log::colors) {
        xmrig::Log::print(GREEN_BOLD(" * ") WHITE_BOLD("COMMANDS     ") MAGENTA_BOLD("h") WHITE_BOLD("ashrate, ")
                                                                     MAGENTA_BOLD("p") WHITE_BOLD("ause, ")
                                                                     MAGENTA_BOLD("r") WHITE_BOLD("esume"));
    }
    else {
        xmrig::Log::print(" * COMMANDS     'h' hashrate, 'p' pause, 'r' resume");
    }
}


void Summary::print(xmrig::Controller *controller)
{
    controller->config()->printVersions();
    print_memory(controller->config());
    print_cpu(controller->config());
    print_threads(controller->config());
    controller->config()->pools().print();

    print_commands(controller->config());
}



