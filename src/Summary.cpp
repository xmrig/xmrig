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
    char buf[16];

#   if defined(__clang__)
    snprintf(buf, 16, " clang/%d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#   elif defined(__GNUC__)
    snprintf(buf, 16, " gcc/%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   elif defined(_MSC_VER)
    snprintf(buf, 16, " MSVC/%d", MSVC_VERSION);
#   else
    buf[0] = '\0';
#   endif


    Log::i()->text(config->isColors() ? "\x1B[01;32m * \x1B[01;37mVERSIONS:     \x1B[01;36mXMRig/%s\x1B[01;37m libuv/%s%s" : " * VERSIONS:     XMRig/%s libuv/%s%s",
                   APP_VERSION, uv_version_string(), buf);
}


static void print_memory(xmrig::Config *config) {
#   ifdef _WIN32
    if (config->isColors()) {
        Log::i()->text("\x1B[01;32m * \x1B[01;37mHUGE PAGES:   %s",
                       Mem::isHugepagesAvailable() ? "\x1B[01;32mavailable" : "\x1B[01;31munavailable");
    }
    else {
        Log::i()->text(" * HUGE PAGES:   %s", Mem::isHugepagesAvailable() ? "available" : "unavailable");
    }
#   endif
}


static void print_cpu(xmrig::Config *config)
{
    if (config->isColors()) {
        Log::i()->text("\x1B[01;32m * \x1B[01;37mCPU:          %s (%d) %sx64 %sAES-NI",
                       Cpu::brand(),
                       Cpu::sockets(),
                       Cpu::isX64() ? "\x1B[01;32m" : "\x1B[01;31m-",
                       Cpu::hasAES() ? "\x1B[01;32m" : "\x1B[01;31m-");
#       ifndef XMRIG_NO_LIBCPUID
        Log::i()->text("\x1B[01;32m * \x1B[01;37mCPU L2/L3:    %.1f MB/%.1f MB", Cpu::l2() / 1024.0, Cpu::l3() / 1024.0);
#       endif
    }
    else {
        Log::i()->text(" * CPU:          %s (%d) %sx64 %sAES-NI", Cpu::brand(), Cpu::sockets(), Cpu::isX64() ? "" : "-", Cpu::hasAES() ? "" : "-");
#       ifndef XMRIG_NO_LIBCPUID
        Log::i()->text(" * CPU L2/L3:    %.1f MB/%.1f MB", Cpu::l2() / 1024.0, Cpu::l3() / 1024.0);
#       endif
    }
}


static void print_threads(xmrig::Config *config)
{
    if (config->threadsMode() != xmrig::Config::Advanced) {
        char buf[32];
        if (config->affinity() != -1L) {
            snprintf(buf, 32, ", affinity=0x%" PRIX64, config->affinity());
        }
        else {
            buf[0] = '\0';
        }

        Log::i()->text(config->isColors() ? "\x1B[01;32m * \x1B[01;37mTHREADS:      \x1B[01;36m%d\x1B[01;37m, %s, av=%d, %sdonate=%d%%%s" : " * THREADS:      %d, %s, av=%d, %sdonate=%d%%%s",
                       config->threadsCount(),
                       config->algoName(),
                       config->algoVariant(),
                       config->isColors() && config->donateLevel() == 0 ? "\x1B[01;31m" : "",
                       config->donateLevel(),
                       buf);
    }
    else {
        Log::i()->text(config->isColors() ? "\x1B[01;32m * \x1B[01;37mTHREADS:      \x1B[01;36m%d\x1B[01;37m, %s, %sdonate=%d%%" : " * THREADS:      %d, %s, %sdonate=%d%%",
                       config->threadsCount(),
                       config->algoName(),
                       config->isColors() && config->donateLevel() == 0 ? "\x1B[01;31m" : "",
                       config->donateLevel());
    }
}


static void print_pools(xmrig::Config *config)
{
    const std::vector<Pool> &pools = config->pools();

    for (size_t i = 0; i < pools.size(); ++i) {
        Log::i()->text(config->isColors() ? "\x1B[01;32m * \x1B[01;37mPOOL #%d:      \x1B[01;36m%s" : " * POOL #%d:      %s",
                       i + 1,
                       pools[i].url()
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

    Log::i()->text(config->isColors() ? "\x1B[01;32m * \x1B[01;37mAPI BIND:     \x1B[01;36m%s:%d" : " * API BIND:     %s:%d",
                   config->isApiIPv6() ? "[::]" : "0.0.0.0", port);
}
#endif


static void print_commands(xmrig::Config *config)
{
    if (config->isColors()) {
        Log::i()->text("\x1B[01;32m * \x1B[01;37mCOMMANDS:     \x1B[01;35mh\x1B[01;37mashrate, \x1B[01;35mp\x1B[01;37mause, \x1B[01;35mr\x1B[01;37mesume");
    }
    else {
        Log::i()->text(" * COMMANDS:     'h' hashrate, 'p' pause, 'r' resume");
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



