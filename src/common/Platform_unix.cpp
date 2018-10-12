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

#ifdef __FreeBSD__
#   include <sys/types.h>
#   include <sys/param.h>
#   include <sys/cpuset.h>
#   include <pthread_np.h>
#endif


#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>
#include <uv.h>


#include "Platform.h"
#include "version.h"

#ifdef XMRIG_NVIDIA_PROJECT
#   include "nvidia/cryptonight.h"
#endif


#ifdef __FreeBSD__
typedef cpuset_t cpu_set_t;
#endif


char *Platform::createUserAgent()
{
    const size_t max = 160;

    char *buf = new char[max];
    int length = snprintf(buf, max, "%s/%s (Linux ", APP_NAME, APP_VERSION);

#   if defined(__x86_64__)
    length += snprintf(buf + length, max - length, "x86_64) libuv/%s", uv_version_string());
#   else
    length += snprintf(buf + length, max - length, "i686) libuv/%s", uv_version_string());
#   endif

#   ifdef XMRIG_NVIDIA_PROJECT
    const int cudaVersion = cuda_get_runtime_version();
    length += snprintf(buf + length, max - length, " CUDA/%d.%d", cudaVersion / 1000, cudaVersion % 100);
#   endif

#   ifdef __GNUC__
    length += snprintf(buf + length, max - length, " gcc/%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   endif

    return buf;
}


bool Platform::setThreadAffinity(uint64_t cpu_id)
{
    cpu_set_t mn;
    CPU_ZERO(&mn);
    CPU_SET(cpu_id, &mn);

#   ifndef __ANDROID__
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mn) == 0;
#   else
    return sched_setaffinity(gettid(), sizeof(cpu_set_t), &mn) == 0;
#   endif
}


void Platform::setProcessPriority(int priority)
{
}


void Platform::setThreadPriority(int priority)
{
    if (priority == -1) {
        return;
    }

    int prio = 19;
    switch (priority)
    {
    case 1:
        prio = 5;
        break;

    case 2:
        prio = 0;
        break;

    case 3:
        prio = -5;
        break;

    case 4:
        prio = -10;
        break;

    case 5:
        prio = -15;
        break;

    default:
        break;
    }

    setpriority(PRIO_PROCESS, 0, prio);

#   ifdef SCHED_IDLE
    if (priority == 0) {
        sched_param param;
        param.sched_priority = 0;

        if (sched_setscheduler(0, SCHED_IDLE, &param) != 0) {
            sched_setscheduler(0, SCHED_BATCH, &param);
        }
    }
#   endif
}
