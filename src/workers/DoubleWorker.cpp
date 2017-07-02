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


#include <thread>


#include "crypto/CryptoNight.h"
#include "workers/DoubleWorker.h"
#include "workers/Workers.h"


DoubleWorker::DoubleWorker(Handle *handle)
    : Worker(handle),
    m_nonce1(0),
    m_nonce2(0)
{
}


void DoubleWorker::start()
{
    while (true) {
        if (Workers::isPaused()) {
            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            while (Workers::isPaused());

            consumeJob();
        }

        while (!Workers::isOutdated(m_sequence)) {
            if ((m_count & 0xF) == 0) {
                storeStats();
            }

            m_count += 2;
            *Job::nonce(m_blob)                = ++m_nonce1;
            *Job::nonce(m_blob + m_job.size()) = ++m_nonce2;

            CryptoNight::hash(m_blob, m_job.size(), m_hash, m_ctx);

            if (*reinterpret_cast<uint64_t*>(m_hash + 24) < m_job.target()) {
                Workers::submit(JobResult(m_job.poolId(), m_job.id(), m_nonce1, m_hash, m_job.diff()));
            }

            if (*reinterpret_cast<uint64_t*>(m_hash + 32 + 24) < m_job.target()) {
                Workers::submit(JobResult(m_job.poolId(), m_job.id(), m_nonce2, m_hash + 32, m_job.diff()));
            }

            std::this_thread::yield();
        }

        consumeJob();
    }
}


void DoubleWorker::consumeJob()
{
    m_job = Workers::job();
    m_sequence = Workers::sequence();

    memcpy(m_blob,                m_job.blob(), m_job.size());
    memcpy(m_blob + m_job.size(), m_job.blob(), m_job.size());

    if (m_job.isNicehash()) {
        m_nonce1 = (*Job::nonce(m_blob)                & 0xff000000U) + (0xffffffU / (m_threads * 2) * m_id);
        m_nonce2 = (*Job::nonce(m_blob + m_job.size()) & 0xff000000U) + (0xffffffU / (m_threads * 2) * (m_id + m_threads));
    }
    else {
        m_nonce1 = 0xffffffffU / (m_threads * 2) * m_id;
        m_nonce2 = 0xffffffffU / (m_threads * 2) * (m_id + m_threads);
    }
}
