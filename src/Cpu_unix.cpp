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


#ifdef __FreeBSD__
#   include <sys/types.h>
#   include <sys/param.h>
#   include <sys/cpuset.h>
#   include <pthread_np.h>
#endif


#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <string.h>


#include "Cpu.h"


#ifdef __FreeBSD__
typedef cpuset_t cpu_set_t;
#endif


void Cpu::init()
{
#   ifdef XMRIG_NO_LIBCPUID
    m_totalThreads = sysconf(_SC_NPROCESSORS_CONF);
#   endif

    initCommon();
}


void Cpu::setAffinity(int id, uint64_t mask)
{
    cpu_set_t set;
    CPU_ZERO(&set);

    for (int i = 0; i < m_totalThreads; i++) {
        if (mask & (1UL << i)) {
            CPU_SET(i, &set);
        }
    }

    if (id == -1) {
#       ifndef __FreeBSD__
        sched_setaffinity(0, sizeof(&set), &set);
#       endif
    } else {
        pthread_setaffinity_np(pthread_self(), sizeof(&set), &set);
    }
}
