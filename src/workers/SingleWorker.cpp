/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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


#include <thread>


#include "crypto/CryptoNight_test.h"
#include "workers/CpuThread.h"
#include "workers/SingleWorker.h"
#include "workers/Workers.h"


SingleWorker::SingleWorker(Handle *handle)
    : Worker(handle)
{
}


bool SingleWorker::start()
{
    if (!selfTest()) {
        return false;
    }

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
            if ((m_count & 0xF) == 0) {
                storeStats();
            }

            m_count++;
            *m_job.nonce() = ++m_result.nonce;

            m_thread->fn(m_job.variant())(m_job.blob(), m_job.size(), m_result.result, m_ctx);
            if (*reinterpret_cast<uint64_t*>(m_result.result + 24) < m_job.target()) {
                Workers::submit(m_result);
            }

            std::this_thread::yield();
        }

        consumeJob();
    }

    return true;
}


bool SingleWorker::resume(const Job &job)
{
    if (m_job.poolId() == -1 && job.poolId() >= 0 && job.id() == m_paused.id()) {
        m_job          = m_paused;
        m_result       = m_job;
        m_result.nonce = *m_job.nonce();
        return true;
    }

    return false;
}


bool SingleWorker::selfTest()
{
    if (m_thread->fn(xmrig::VARIANT_NONE) == nullptr) {
        return false;
    }

    m_thread->fn(xmrig::VARIANT_NONE)(test_input, 76, m_result.result, m_ctx);

    if (m_thread->algorithm() == xmrig::CRYPTONIGHT && memcmp(m_result.result, test_output_v0, 32) == 0) {
        m_thread->fn(xmrig::VARIANT_V1)(test_input, 76, m_result.result, m_ctx);

        return memcmp(m_result.result, test_output_v1, 32) == 0;
    }

#   ifndef XMRIG_NO_AEON
    if (m_thread->algorithm() == xmrig::CRYPTONIGHT_LITE && memcmp(m_result.result, test_output_v0_lite, 32) == 0) {
        m_thread->fn(xmrig::VARIANT_V1)(test_input, 76, m_result.result, m_ctx);

        return memcmp(m_result.result, test_output_v1_lite, 32) == 0;
    }
#   endif

    return memcmp(m_result.result, test_output_heavy, 32) == 0;
}


void SingleWorker::consumeJob()
{
    Job job = Workers::job();
    m_sequence = Workers::sequence();
    if (m_job == job) {
        return;
    }

    save(job);

    if (resume(job)) {
        return;
    }

    m_job = std::move(job);
    m_result = m_job;

    if (m_job.isNicehash()) {
        m_result.nonce = (*m_job.nonce() & 0xff000000U) + (0xffffffU / m_totalWays * m_id);
    }
    else {
        m_result.nonce = 0xffffffffU / m_totalWays * m_id;
    }
}


void SingleWorker::save(const Job &job)
{
    if (job.poolId() == -1 && m_job.poolId() >= 0) {
        m_paused = m_job;
    }
}
