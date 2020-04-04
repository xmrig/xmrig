/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "backend/cuda/CudaWorker.h"
#include "backend/common/Tags.h"
#include "backend/cuda/runners/CudaCnRunner.h"
#include "base/io/log/Log.h"
#include "base/tools/Chrono.h"
#include "core/Miner.h"
#include "crypto/common/Nonce.h"
#include "net/JobResults.h"


#ifdef XMRIG_ALGO_RANDOMX
#   include "backend/cuda/runners/CudaRxRunner.h"
#endif


#ifdef XMRIG_ALGO_ASTROBWT
#   include "backend/cuda/runners/CudaAstroBWTRunner.h"
#endif


#include <cassert>
#include <thread>


namespace xmrig {


static constexpr uint32_t kReserveCount = 32768;
std::atomic<bool> CudaWorker::ready;


static inline bool isReady()                         { return !Nonce::isPaused() && CudaWorker::ready; }
static inline uint32_t roundSize(uint32_t intensity) { return kReserveCount / intensity + 1; }


} // namespace xmrig



xmrig::CudaWorker::CudaWorker(size_t id, const CudaLaunchData &data) :
    Worker(id, data.thread.affinity(), -1),
    m_algorithm(data.algorithm),
    m_miner(data.miner)
{
    switch (m_algorithm.family()) {
    case Algorithm::RANDOM_X:
#       ifdef XMRIG_ALGO_RANDOMX
        m_runner = new CudaRxRunner(id, data);
#       endif
        break;

    case Algorithm::ARGON2:
        break;

    case Algorithm::ASTROBWT:
#       ifdef XMRIG_ALGO_ASTROBWT
        m_runner = new CudaAstroBWTRunner(id, data);
#       endif
        break;

    default:
        m_runner = new CudaCnRunner(id, data);
        break;
    }

    if (!m_runner) {
        return;
    }

    if (!m_runner->init()) {
        delete m_runner;

        m_runner = nullptr;
    }
}


xmrig::CudaWorker::~CudaWorker()
{
    delete m_runner;
}


bool xmrig::CudaWorker::selfTest()
{
    return m_runner != nullptr;
}


size_t xmrig::CudaWorker::intensity() const
{
    return m_runner ? m_runner->roundSize() : 0;
}


void xmrig::CudaWorker::start()
{
    while (Nonce::sequence(Nonce::CUDA) > 0) {
        if (!isReady()) {
            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            while (!isReady() && Nonce::sequence(Nonce::CUDA) > 0);

            if (Nonce::sequence(Nonce::CUDA) == 0) {
                break;
            }

            if (!consumeJob()) {
                return;
            }
        }

        while (!Nonce::isOutdated(Nonce::CUDA, m_job.sequence())) {
            uint32_t foundNonce[10] = { 0 };
            uint32_t foundCount     = 0;

            if (!m_runner->run(*m_job.nonce(), &foundCount, foundNonce)) {
                return;
            }

            if (foundCount) {
                JobResults::submit(m_job.currentJob(), foundNonce, foundCount);
            }

            const size_t batch_size = intensity();
            if (!m_job.nextRound(roundSize(batch_size), batch_size)) {
                JobResults::done(m_job.currentJob());
            }

            storeStats();
            std::this_thread::yield();
        }

        if (!consumeJob()) {
            return;
        }
    }
}


bool xmrig::CudaWorker::consumeJob()
{
    if (Nonce::sequence(Nonce::CUDA) == 0) {
        return false;
    }

    const size_t batch_size = intensity();
    m_job.add(m_miner->job(), roundSize(batch_size) * batch_size, Nonce::CUDA);

    return m_runner->set(m_job.currentJob(), m_job.blob());;
}


void xmrig::CudaWorker::storeStats()
{
    if (!isReady()) {
        return;
    }

    m_count += m_runner ? m_runner->processedHashes() : 0;

    Worker::storeStats();
}
