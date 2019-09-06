/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "backend/opencl/OclWorker.h"

#include "backend/opencl/runners/OclCnRunner.h"
#include "base/io/log/Log.h"
#include "base/tools/Chrono.h"
#include "core/Miner.h"
#include "crypto/common/Nonce.h"
#include "net/JobResults.h"


#ifdef XMRIG_ALGO_RANDOMX
#   include "backend/opencl/runners/OclRxRunner.h"
#endif


#include <cassert>
#include <thread>


namespace xmrig {


static constexpr uint32_t kReserveCount = 32768;
std::atomic<bool> OclWorker::ready;


static inline bool isReady()                         { return !Nonce::isPaused() && OclWorker::ready; }
static inline uint32_t roundSize(uint32_t intensity) { return kReserveCount / intensity + 1; }


} // namespace xmrig



xmrig::OclWorker::OclWorker(size_t id, const OclLaunchData &data) :
    Worker(id, data.affinity, -1),
    m_algorithm(data.algorithm),
    m_miner(data.miner),
    m_intensity(data.thread.intensity()),
    m_interleave(data.interleave)
{
    switch (m_algorithm.family()) {
    case Algorithm::RANDOM_X:
#       ifdef XMRIG_ALGO_RANDOMX
        m_runner = new OclRxRunner(id, data);
#       endif
        break;

    case Algorithm::ARGON2:
#       ifdef XMRIG_ALGO_ARGON2
        m_runner = nullptr; // TODO OclArgon2Runner
#       endif
        break;

    default:
        m_runner = new OclCnRunner(id, data);
        break;
    }

    if (m_runner) {
        m_runner->build();
    }
}


xmrig::OclWorker::~OclWorker()
{
    delete m_runner;
}


bool xmrig::OclWorker::selfTest()
{
    return m_runner && m_runner->selfTest();
}


void xmrig::OclWorker::start()
{
    cl_uint results[0x100];

    while (Nonce::sequence(Nonce::OPENCL) > 0) {
        if (!isReady()) {
            if (m_interleave) {
                m_interleave->setResumeCounter(0);
            }

            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            while (!isReady() && Nonce::sequence(Nonce::OPENCL) > 0);

            if (Nonce::sequence(Nonce::OPENCL) == 0) {
                break;
            }

            if (m_interleave) {
                m_interleave->resumeDelay(m_id);
            }

            consumeJob();
        }

        while (!Nonce::isOutdated(Nonce::OPENCL, m_job.sequence())) {
            if (m_interleave) {
                m_interleave->adjustDelay(m_id);
            }

            const uint64_t t = Chrono::steadyMSecs();

            if (!m_runner->run(*m_job.nonce(), results)) {
                return;
            }

            if (results[0xFF] > 0) {
                JobResults::submit(m_job.currentJob(), results, results[0xFF]);
            }

            m_job.nextRound(roundSize(m_intensity), m_intensity);

            storeStats(t);
            std::this_thread::yield();
        }

        consumeJob();
    }
}


void xmrig::OclWorker::consumeJob()
{
    if (Nonce::sequence(Nonce::OPENCL) == 0) {
        return;
    }

    m_job.add(m_miner->job(), Nonce::sequence(Nonce::OPENCL), roundSize(m_intensity) * m_intensity);
    m_runner->set(m_job.currentJob(), m_job.blob());
}


void xmrig::OclWorker::storeStats(uint64_t t)
{
    if (!isReady()) {
        return;
    }

    m_count += m_intensity;

    if (m_interleave) {
        m_interleave->setRunTime(Chrono::steadyMSecs() - t);
    }

    Worker::storeStats();
}
