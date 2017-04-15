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

#include <stdlib.h>
#include <signal.h>
#include <errno.h>
#include <unistd.h>

#include "options.h"
#include "cpu.h"
#include "utils/applog.h"


static void signal_handler(int sig)
{
    switch (sig) {
    case SIGHUP:
        applog(LOG_WARNING, "SIGHUP received");
        break;

    case SIGINT:
        applog(LOG_WARNING, "SIGINT received, exiting");
        proper_exit(0);
        break;

    case SIGTERM:
        applog(LOG_WARNING, "SIGTERM received, exiting");
        proper_exit(0);
        break;
    }
}


void proper_exit(int reason) {
    exit(reason);
}


void os_specific_init()
{
    if (opt_affinity != -1) {
        affine_to_cpu_mask(-1, opt_affinity);
    }

    if (opt_background) {
        int i = fork();
        if (i < 0) {
            exit(1);
        }

        if (i > 0) {
            exit(0);
        }

        i = setsid();

        if (i < 0) {
            applog(LOG_ERR, "setsid() failed (errno = %d)", errno);
        }

        i = chdir("/");
        if (i < 0) {
            applog(LOG_ERR, "chdir() failed (errno = %d)", errno);
        }

        signal(SIGHUP, signal_handler);
        signal(SIGTERM, signal_handler);
    }

    signal(SIGINT, signal_handler);
}
