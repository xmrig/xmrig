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


class DoubleWorker::State
{
public:
  inline State() :
      nonce1(0),
      nonce2(0)
  {}

  Job job;
  uint32_t nonce1;
  uint32_t nonce2;
  uint8_t blob[84 * 2];
};


DoubleWorker::DoubleWorker(Handle *handle)
    : Worker(handle)
{
    m_state       = new State();
    m_pausedState = new State();
}


DoubleWorker::~DoubleWorker()
{
    delete m_state;
    delete m_pausedState;
}


void DoubleWorker::start()
{
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

            m_count += 2;
            *Job::nonce(m_state->blob)                       = ++m_state->nonce1;
            *Job::nonce(m_state->blob + m_state->job.size()) = ++m_state->nonce2;

            CryptoNight::hash(m_state->blob, m_state->job.size(), m_hash, m_ctx);

            if (*reinterpret_cast<uint64_t*>(m_hash + 24) < m_state->job.target()) {
                Workers::submit(JobResult(m_state->job.poolId(), m_state->job.id(), m_state->nonce1, m_hash, m_state->job.diff()));
            }

            if (*reinterpret_cast<uint64_t*>(m_hash + 32 + 24) < m_state->job.target()) {
                Workers::submit(JobResult(m_state->job.poolId(), m_state->job.id(), m_state->nonce2, m_hash + 32, m_state->job.diff()));
            }

            std::this_thread::yield();
        }

        consumeJob();
    }
}


bool DoubleWorker::resume(const Job &job)
{
    if (m_state->job.poolId() == -1 && job.poolId() >= 0 && job.id() == m_pausedState->job.id()) {
        *m_state = *m_pausedState;
        return true;
    }

    return false;
}


void DoubleWorker::consumeJob()
{
    Job job = Workers::job();
    m_sequence = Workers::sequence();
    if (m_state->job == job) {
        return;
    }

    save(job);

    if (resume(job)) {
        return;
    }

    m_state->job = std::move(job);
    memcpy(m_state->blob,                       m_state->job.blob(), m_state->job.size());
    memcpy(m_state->blob + m_state->job.size(), m_state->job.blob(), m_state->job.size());

    if (m_state->job.isNicehash()) {
        m_state->nonce1 = (*Job::nonce(m_state->blob)                       & 0xff000000U) + (0xffffffU / (m_threads * 2) * m_id);
        m_state->nonce2 = (*Job::nonce(m_state->blob + m_state->job.size()) & 0xff000000U) + (0xffffffU / (m_threads * 2) * (m_id + m_threads));
    }
    else {
        m_state->nonce1 = 0xffffffffU / (m_threads * 2) * m_id;
        m_state->nonce2 = 0xffffffffU / (m_threads * 2) * (m_id + m_threads);
    }
}


void DoubleWorker::save(const Job &job)
{
    if (job.poolId() == -1 && m_state->job.poolId() >= 0) {
        *m_pausedState = *m_state;
    }
}
