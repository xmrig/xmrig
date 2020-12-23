/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <support@xmrig.com>
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


#include <cinttypes>
#include <cstdio>
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


#ifdef XMRIG_ALGO_RANDOMX
#   include "crypto/rx/RxConfig.h"
#endif


namespace xmrig {


#ifdef XMRIG_OS_WIN
static constexpr const char *kHugepagesSupported = GREEN_BOLD("permission granted");
#else
static constexpr const char *kHugepagesSupported = GREEN_BOLD("supported");
#endif


#ifdef XMRIG_FEATURE_ASM
static const char *coloredAsmNames[] = {
    RED_BOLD("none"),
    "auto",
    GREEN_BOLD("intel"),
    GREEN_BOLD("ryzen"),
    GREEN_BOLD("bulldozer")
};


inline static const char *asmName(Assembly::Id assembly)
{
    return coloredAsmNames[assembly];
}
#endif


static void print_memory(Config *config)
{
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") "%s",
               "HUGE PAGES", config->cpu().isHugePages() ? (VirtualMemory::isHugepagesAvailable() ? kHugepagesSupported : RED_BOLD("unavailable")) : RED_BOLD("disabled"));

#   ifdef XMRIG_ALGO_RANDOMX
#   ifdef XMRIG_OS_LINUX
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") "%s",
               "1GB PAGES", (VirtualMemory::isOneGbPagesAvailable() ? (config->rx().isOneGbPages() ? kHugepagesSupported : YELLOW_BOLD("disabled")) : YELLOW_BOLD("unavailable")));
#   else
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") "%s", "1GB PAGES", YELLOW_BOLD("unavailable"));
#   endif
#   endif
}


static void print_cpu(Config *)
{
    const auto info = Cpu::info();

    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s%s (%zu)") " %s %sAES%s",
               "CPU",
               info->brand(),
               info->packages(),
               ICpuInfo::is64bit()    ? GREEN_BOLD("64-bit") : RED_BOLD("32-bit"),
               info->hasAES()         ? GREEN_BOLD_S : RED_BOLD_S "-",
               info->isVM()           ? RED_BOLD_S " VM" : ""
               );
#   if defined(XMRIG_FEATURE_HWLOC)
    Log::print(WHITE_BOLD("   %-13s") BLACK_BOLD("L2:") WHITE_BOLD("%.1f MB") BLACK_BOLD(" L3:") WHITE_BOLD("%.1f MB")
               CYAN_BOLD(" %zu") "C" BLACK_BOLD("/") CYAN_BOLD("%zu") "T"
               BLACK_BOLD(" NUMA:") CYAN_BOLD("%zu"),
               "",
               info->L2() / 1048576.0,
               info->L3() / 1048576.0,
               info->cores(),
               info->threads(),
               info->nodes()
               );
#   else
    Log::print(WHITE_BOLD("   %-13s") BLACK_BOLD("threads:") CYAN_BOLD("%zu"), "", info->threads());
#   endif
}


static void print_memory()
{
    constexpr size_t oneGiB = 1024U * 1024U * 1024U;
    const auto freeMem      = static_cast<double>(uv_get_free_memory());
    const auto totalMem     = static_cast<double>(uv_get_total_memory());

    const double percent = freeMem > 0 ? ((totalMem - freeMem) / totalMem * 100.0) : 100.0;

    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("%.1f/%.1f GB") BLACK_BOLD(" (%.0f%%)"),
               "MEMORY",
               (totalMem - freeMem) / oneGiB,
               totalMem / oneGiB,
               percent
               );
}


static void print_threads(Config *config)
{
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") WHITE_BOLD("%s%d%%"),
               "DONATE",
               config->pools().donateLevel() == 0 ? RED_BOLD_S : "",
               config->pools().donateLevel()
               );

#   ifdef XMRIG_FEATURE_ASM
    if (config->cpu().assembly() == Assembly::AUTO) {
        const Assembly assembly = Cpu::info()->assembly();

        Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13sauto:%s"), "ASSEMBLY", asmName(assembly));
    }
    else {
        Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s%s"), "ASSEMBLY", asmName(config->cpu().assembly()));
    }
#   endif
}


static void print_commands(Config *)
{
    if (Log::isColors()) {
        Log::print(GREEN_BOLD(" * ") WHITE_BOLD("COMMANDS     ") MAGENTA_BG_BOLD("h") WHITE_BOLD("ashrate, ")
                                                                 MAGENTA_BG_BOLD("p") WHITE_BOLD("ause, ")
                                                                 MAGENTA_BG_BOLD("r") WHITE_BOLD("esume, ")
                                                                 WHITE_BOLD("re") MAGENTA_BG(WHITE_BOLD_S "s") WHITE_BOLD("ults, ")
                                                                 MAGENTA_BG_BOLD("c") WHITE_BOLD("onnection")
                   );
    }
    else {
        Log::print(" * COMMANDS     'h' hashrate, 'p' pause, 'r' resume, 's' results, 'c' connection");
    }
}


} // namespace xmrig


void xmrig::Summary::print(Controller *controller)
{
    controller->config()->printVersions();
    print_memory(controller->config());
    print_cpu(controller->config());
    print_memory();
    print_threads(controller->config());
    controller->config()->pools().print();

    print_commands(controller->config());
}



