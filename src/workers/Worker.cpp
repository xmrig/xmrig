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

#include <chrono>


#include "Cpu.h"
#include "Mem.h"
#include "Platform.h"
#include "workers/Handle.h"
#include "workers/Worker.h"


Worker::Worker(Handle *handle) :
    m_id(handle->threadId()),
    m_threads(handle->threads()),
    m_hashCount(0),
    m_timestamp(0),
    m_count(0),
    m_sequence(0)
{
    if (Cpu::threads() > 1 && handle->affinity() != -1L) {
        Cpu::setAffinity(m_id, handle->affinity());
    }

    Platform::setThreadPriority(handle->priority());
    m_ctx = Mem::create(m_id);
}


Worker::~Worker()
{
}


void Worker::storeStats()
{
    using namespace std::chrono;

    const uint64_t timestamp = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();
    m_hashCount.store(m_count, std::memory_order_relaxed);
    m_timestamp.store(timestamp, std::memory_order_relaxed);
}
