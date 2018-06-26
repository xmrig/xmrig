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


#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>


#include "CpuImpl.h"
#include "Cpu.h"

void CpuImpl::init()
{
#   ifdef XMRIG_NO_LIBCPUID
    m_totalThreads = sysconf(_SC_NPROCESSORS_CONF);
#   endif

    initCommon();
}

int CpuImpl::setThreadAffinity(size_t threadId, int64_t affinityMask)
{
    int cpuId = -1;

    if (affinityMask != -1L) {
        cpuId = Cpu::getAssignedCpuId(threadId, affinityMask);
    } else {
        cpuId = static_cast<int>(threadId);
    }

    if (cpuId > -1) {
        thread_port_t mach_thread;
        thread_affinity_policy_data_t policy = {static_cast<integer_t>(cpuId)};
        mach_thread = pthread_mach_thread_np(pthread_self());

        thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, (thread_policy_t) & policy, 1);
    }

    return cpuId;
}
