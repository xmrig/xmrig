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


#include "backend/opencl/OclWorker.h"
#include "backend/common/Tags.h"
#include "backend/opencl/runners/OclCnRunner.h"
#include "backend/opencl/runners/tools/OclSharedData.h"
#include "backend/opencl/runners/tools/OclSharedState.h"
#include "base/io/log/Log.h"
#include "base/tools/Chrono.h"
#include "core/Miner.h"
#include "crypto/common/Nonce.h"
#include "net/JobResults.h"


#ifdef XMRIG_ALGO_RANDOMX
#   include "backend/opencl/runners/OclRxJitRunner.h"
#   include "backend/opencl/runners/OclRxVmRunner.h"
#endif

#ifdef XMRIG_ALGO_ASTROBWT
#   include "backend/opencl/runners/OclAstroBWTRunner.h"
#endif

#ifdef XMRIG_ALGO_CN_GPU
#   include "backend/opencl/runners/OclRyoRunner.h"
#endif


#include <cassert>
#include <thread>


namespace xmrig {


static constexpr uint32_t kReserveCount = 32768;
std::atomic<bool> OclWorker::ready;


static inline bool isReady()                         { return !Nonce::isPaused() && OclWorker::ready; }
static inline uint32_t roundSize(uint32_t intensity) { return kReserveCount / intensity + 1; }


static inline void printError(size_t id, const char *error)
{
    LOG_ERR("%s" RED_S " thread " RED_BOLD("#%zu") RED_S " failed with error " RED_BOLD("%s"), ocl_tag(), id, error);
}


} // namespace xmrig



xmrig::OclWorker::OclWorker(size_t id, const OclLaunchData &data) :
    Worker(id, data.affinity, -1),
    m_algorithm(data.algorithm),
    m_miner(data.miner),
    m_intensity(data.thread.intensity()),
    m_sharedData(OclSharedState::get(data.device.index()))
{
    switch (m_algorithm.family()) {
    case Algorithm::RANDOM_X:
#       ifdef XMRIG_ALGO_RANDOMX
        if (data.thread.isAsm() && data.device.vendorId() == OCL_VENDOR_AMD) {
            m_runner = new OclRxJitRunner(id, data);
        }
        else {
            m_runner = new OclRxVmRunner(id, data);
        }
#       endif
        break;

    case Algorithm::ARGON2:
#       ifdef XMRIG_ALGO_ARGON2
        m_runner = nullptr; // TODO OclArgon2Runner
#       endif
        break;

    case Algorithm::ASTROBWT:
#       ifdef XMRIG_ALGO_ASTROBWT
        m_runner = new OclAstroBWTRunner(id, data);
#       endif
        break;

    default:
#       ifdef XMRIG_ALGO_CN_GPU
        if (m_algorithm == Algorithm::CN_GPU) {
            m_runner = new OclRyoRunner(id, data);
        }
        else
#       endif
        {
            m_runner = new OclCnRunner(id, data);
        }
        break;
    }

    if (!m_runner) {
        return;
    }

    try {
        m_runner->init();
        m_runner->build();
    }
    catch (std::exception &ex) {
        printError(id, ex.what());

        delete m_runner;
        m_runner = nullptr;
    }
}


xmrig::OclWorker::~OclWorker()
{
    delete m_runner;
}


bool xmrig::OclWorker::selfTest()
{
    return m_runner != nullptr;
}


size_t xmrig::OclWorker::intensity() const
{
    return m_runner ? m_runner->intensity() : 0;
}


void xmrig::OclWorker::start()
{
    cl_uint results[0x100];

    const uint32_t runnerRoundSize = m_runner->roundSize();

    while (Nonce::sequence(Nonce::OPENCL) > 0) {
        if (!isReady()) {
            m_sharedData.setResumeCounter(0);

            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            while (!isReady() && Nonce::sequence(Nonce::OPENCL) > 0);

            if (Nonce::sequence(Nonce::OPENCL) == 0) {
                break;
            }

            m_sharedData.resumeDelay(m_id);

            if (!consumeJob()) {
                return;
            }
        }

        while (!Nonce::isOutdated(Nonce::OPENCL, m_job.sequence())) {
            m_sharedData.adjustDelay(m_id);

            const uint64_t t = Chrono::steadyMSecs();

            try {
                m_runner->run(*m_job.nonce(), results);
            }
            catch (std::exception &ex) {
                printError(id(), ex.what());

                return;
            }

            if (results[0xFF] > 0) {
                JobResults::submit(m_job.currentJob(), results, results[0xFF]);
            }

            if (!m_job.nextRound(roundSize(runnerRoundSize), runnerRoundSize)) {
                JobResults::done(m_job.currentJob());
            }

            storeStats(t);
            std::this_thread::yield();
        }

        if (!consumeJob()) {
            return;
        }
    }
}


bool xmrig::OclWorker::consumeJob()
{
    if (Nonce::sequence(Nonce::OPENCL) == 0) {
        return false;
    }

    m_job.add(m_miner->job(), roundSize(m_intensity) * m_intensity, Nonce::OPENCL);

    try {
        m_runner->set(m_job.currentJob(), m_job.blob());
    }
    catch (std::exception &ex) {
        printError(id(), ex.what());

        return false;
    }

    return true;
}


void xmrig::OclWorker::storeStats(uint64_t t)
{
    if (!isReady()) {
        return;
    }

    m_count += m_runner->processedHashes();

    m_sharedData.setRunTime(Chrono::steadyMSecs() - t);

    Worker::storeStats();
}
