/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "common/log/Log.h"
#include "common/net/Pool.h"
#include "core/Config.h"
#include "core/Controller.h"
#include "Cpu.h"
#include "Mem.h"
#include "Summary.h"
#include "version.h"


static void print_versions(xmrig::Config *config)
{
    char buf[16] = { 0 };

#   if defined(__clang__)
    snprintf(buf, 16, " clang/%d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#   elif defined(__GNUC__)
    snprintf(buf, 16, " gcc/%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   elif defined(_MSC_VER)
    snprintf(buf, 16, " MSVC/%d", MSVC_VERSION);
#   endif

    Log::i()->text(config->isColors() ? GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("%s/%s") WHITE_BOLD(" libuv/%s%s")
                                      : " * %-13s%s/%s libuv/%s%s",
                   "VERSIONS", APP_NAME, APP_VERSION, uv_version_string(), buf);
}


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
    if (config->isColors()) {
        Log::i()->text(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") "%s (%d) %sx64 %sAES-NI",
                       "CPU",
                       Cpu::brand(),
                       Cpu::sockets(),
                       Cpu::isX64() ? "\x1B[1;32m" : "\x1B[1;31m-",
                       Cpu::hasAES() ? "\x1B[1;32m" : "\x1B[1;31m-");
#       ifndef XMRIG_NO_LIBCPUID
        Log::i()->text(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") "%.1f MB/%.1f MB", "CPU L2/L3", Cpu::l2() / 1024.0, Cpu::l3() / 1024.0);
#       endif
    }
    else {
        Log::i()->text(" * %-13s%s (%d) %sx64 %sAES-NI", "CPU", Cpu::brand(), Cpu::sockets(), Cpu::isX64() ? "" : "-", Cpu::hasAES() ? "" : "-");
#       ifndef XMRIG_NO_LIBCPUID
        Log::i()->text(" * %-13s%.1f MB/%.1f MB", "CPU L2/L3", Cpu::l2() / 1024.0, Cpu::l3() / 1024.0);
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
}


static void print_pools(xmrig::Config *config)
{
    const std::vector<Pool> &pools = config->pools();

    for (size_t i = 0; i < pools.size(); ++i) {
        Log::i()->text(config->isColors() ? GREEN_BOLD(" * ") WHITE_BOLD("POOL #%-7zu") CYAN_BOLD("%s") " variant " WHITE_BOLD("%s")
                                          : " * POOL #%-7d%s variant %s",
                       i + 1,
                       pools[i].url(),
                       pools[i].algorithm().variantName()
                       );
    }

#   ifdef APP_DEBUG
    for (const Pool &pool : pools) {
        pool.print();
    }
#   endif
}


#ifndef XMRIG_NO_API
static void print_api(xmrig::Config *config)
{
    const int port = config->apiPort();
    if (port == 0) {
        return;
    }

    Log::i()->text(config->isColors() ? GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN("%s:") CYAN_BOLD("%d")
                                      : " * %-13s%s:%d",
                   "API BIND", config->isApiIPv6() ? "[::]" : "0.0.0.0", port);
}
#endif


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
    print_versions(controller->config());
    print_memory(controller->config());
    print_cpu(controller->config());
    print_threads(controller->config());
    print_pools(controller->config());

#   ifndef XMRIG_NO_API
    print_api(controller->config());
#   endif

    print_commands(controller->config());
}



