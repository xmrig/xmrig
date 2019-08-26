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
#include "Summary.h"
#include "version.h"

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

static void print_donate(xmrig::Config *config)
{
    if (config->isColors()) {
        Log::i()->text(GREEN_BOLD(" * ") WHITE_BOLD("DONATE       ") MAGENTA_BOLD("%d%%") WHITE_BOLD(" (change with --donate-level option)"), config->donateLevel());
    }
    else {
        Log::i()->text(" * DONATE       %d%% (change with --donate-level option)", config->donateLevel());
    }
}

void Summary::print(xmrig::Controller *controller)
{
    controller->config()->printVersions();
    controller->config()->printPools();
    controller->config()->printAPI();

    print_donate(controller->config());

    print_commands(controller->config());
}



