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

#include <chrono>


#include "common/cpu/Cpu.h"
#include "common/Platform.h"
#include "core/HasherConfig.h"
#include "workers/Handle.h"
#include "workers/Worker.h"
#include "workers/Workers.h"


Worker::Worker(Handle *handle, int workerIdx) :
        m_id(workerIdx),
        m_hashCount(0),
        m_timestamp(0),
        m_count(0),
        m_sequence(0),
        m_config(static_cast<xmrig::HasherConfig *>(handle->config())),
        m_hasher(handle->hasher())
{
    m_offset = handle->offset() + m_id;
    m_hash = new uint8_t[m_hasher->parallelism(m_id) * 36];
}

void Worker::storeStats()
{
    using namespace std::chrono;

    const uint64_t timestamp = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();
    m_hashCount.store(m_count, std::memory_order_relaxed);
    m_timestamp.store(timestamp, std::memory_order_relaxed);
}

bool Worker::selfTest()
{
    return true;
}

void Worker::start() {
    if(m_hasher->type() == "CPU" && m_hasher->subType() == "CPU") {
        if (xmrig::Cpu::info()->threads() > 1 && m_config->getCPUAffinity(m_id) != -1L) {
            Platform::setThreadAffinity(m_config->getCPUAffinity(m_id));
        }
    }

    Platform::setThreadPriority(m_config->priority());
    int parallelism = m_hasher->parallelism(m_id);

    while (Workers::sequence() > 0) {
        if (Workers::isPaused()) {
            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            while (Workers::isPaused());

            if (Workers::sequence() == 0) {
                break;
            }

            consumeJob();
        }

        while (!Workers::isOutdated(m_sequence)) {
            int hashCount = m_hasher->compute(m_id, m_state.blob, m_state.job.size(), m_hash);

            if(hashCount == parallelism) {

                for (size_t i = 0; i < parallelism; ++i) {
                    if (*reinterpret_cast<uint64_t *>(m_hash + (i * 36) + 24) < m_state.job.target()) {
                        Workers::submit(xmrig::JobResult(m_state.job.poolId(), m_state.job.id(), m_state.job.clientId(),
                                                         *reinterpret_cast<uint32_t*>(m_hash + (i * 36) + 32), m_hash + (i * 36), m_state.job.diff(),
                                                         m_state.job.algorithm()));
                    }
                }

                m_count += parallelism;
            }

            storeStats();

            std::this_thread::yield();
        }

        consumeJob();
    }
}

bool Worker::consumeJob() {
    xmrig::Job job = Workers::job();
    m_sequence = Workers::sequence();
    if (m_state.job == job) {
        return false;
    }

    save(job);

    if (resume(job)) {
        return false;
    }

    m_state.job = job;

    const size_t size = m_state.job.size();
    memcpy(m_state.blob, m_state.job.blob(), size);

    uint32_t *nonce = reinterpret_cast<uint32_t*>(m_state.blob + 39);
    if (m_state.job.isNicehash()) {
        *nonce = (*nonce & 0xff000000U) + (0xffffffU / Workers::totalThreads() * m_offset);
    }
    else {
        *nonce = 0xffffffffU / Workers::totalThreads() * m_offset;
    }

    return true;
}

bool Worker::resume(const xmrig::Job &job)
{
    if (m_state.job.poolId() == -1 && job.poolId() >= 0 && job.id() == m_pausedState.job.id()) {
        m_state = m_pausedState;
        return true;
    }

    return false;
}

void Worker::save(const xmrig::Job &job)
{
    if (job.poolId() == -1 && m_state.job.poolId() >= 0) {
        m_pausedState = m_state;
    }
}
