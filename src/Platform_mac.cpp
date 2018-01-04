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


#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <uv.h>


#include "Platform.h"
#include "version.h"

#ifdef XMRIG_NVIDIA_PROJECT
#   include "nvidia/cryptonight.h"
#endif


static inline char *createUserAgent()
{
    const size_t max = 160;

    char *buf = new char[max];

#   ifdef XMRIG_NVIDIA_PROJECT
    const int cudaVersion = cuda_get_runtime_version();
    snprintf(buf, max, "%s/%s (Macintosh; Intel Mac OS X) libuv/%s CUDA/%d.%d clang/%d.%d.%d", APP_NAME, APP_VERSION, uv_version_string(), cudaVersion / 1000, cudaVersion % 100, __clang_major__, __clang_minor__, __clang_patchlevel__);
#   else
    snprintf(buf, max, "%s/%s (Macintosh; Intel Mac OS X) libuv/%s clang/%d.%d.%d", APP_NAME, APP_VERSION, uv_version_string(), __clang_major__, __clang_minor__, __clang_patchlevel__);
#   endif

    return buf;
}


void Platform::init(const char *userAgent)
{
    m_userAgent = userAgent ? strdup(userAgent) : createUserAgent();
}


void Platform::release()
{
    delete [] m_userAgent;
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
}

