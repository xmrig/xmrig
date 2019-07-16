/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#include <cmath>
#include <inttypes.h>
#include <thread>


#include "api/Api.h"
#include "backend/cpu/CpuWorker.h"
#include "base/io/log/Log.h"
#include "base/tools/Chrono.h"
#include "base/tools/Handle.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "crypto/common/Nonce.h"
#include "crypto/rx/RxAlgo.h"
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxDataset.h"
#include "Mem.h"
#include "rapidjson/document.h"
//#include "workers/Hashrate.h"
#include "workers/WorkersLegacy.h"


bool WorkersLegacy::m_active = false;
bool WorkersLegacy::m_enabled = true;
//Hashrate *WorkersLegacy::m_hashrate = nullptr;
xmrig::Job WorkersLegacy::m_job;
WorkersLegacy::LaunchStatus WorkersLegacy::m_status;
std::vector<xmrig::Thread<xmrig::CpuLaunchData>* > WorkersLegacy::m_workers;
uint64_t WorkersLegacy::m_ticks = 0;
uv_mutex_t WorkersLegacy::m_mutex;
uv_rwlock_t WorkersLegacy::m_rwlock;
//uv_timer_t *Workers::m_timer = nullptr;
xmrig::Controller *WorkersLegacy::m_controller = nullptr;


//xmrig::Job WorkersLegacy::job()
//{
//    uv_rwlock_rdlock(&m_rwlock);
//    xmrig::Job job = m_job;
//    uv_rwlock_rdunlock(&m_rwlock);

//    return job;
//}


//size_t WorkersLegacy::hugePages()
//{
//    uv_mutex_lock(&m_mutex);
//    const size_t hugePages = m_status.hugePages;
//    uv_mutex_unlock(&m_mutex);

//    return hugePages;
//}


//size_t WorkersLegacy::threads()
//{
//    uv_mutex_lock(&m_mutex);
//    const size_t threads = m_status.threads;
//    uv_mutex_unlock(&m_mutex);

//    return threads;
//}


//void Workers::pause()
//{
//    m_active = false;

//    xmrig::Nonce::pause(true);
//    xmrig::Nonce::touch();
//}


//void Workers::setEnabled(bool enabled)
//{
//    if (m_enabled == enabled) {
//        return;
//    }

//    m_enabled = enabled;
//    if (!m_active) {
//        return;
//    }

//    xmrig::Nonce::pause(!enabled);
//    xmrig::Nonce::touch();
//}


//void Workers::setJob(const xmrig::Job &job, bool donate)
//{
//    uv_rwlock_wrlock(&m_rwlock);

//    m_job = job;
//    m_job.setIndex(donate ? 1 : 0);

//    xmrig::Nonce::reset(donate ? 1 : 0);

//    uv_rwlock_wrunlock(&m_rwlock);

//    m_active = true;
//    if (!m_enabled) {
//        return;
//    }

//    xmrig::Nonce::pause(false);
//}


void WorkersLegacy::start(xmrig::Controller *controller)
{
    using namespace xmrig;

#   ifdef APP_DEBUG
    LOG_NOTICE("THREADS ------------------------------------------------------------------");
    for (const xmrig::IThread *thread : controller->config()->threads()) {
        thread->print();
    }
    LOG_NOTICE("--------------------------------------------------------------------------");
#   endif

    m_controller = controller;

    m_status.algo               = xmrig::Algorithm::RX_WOW; // FIXME algo
    const CpuThreads &threads   = controller->config()->cpu().threads().get(m_status.algo);
    m_status.threads            = threads.size();

    for (const CpuThread &thread : threads) {
       m_status.ways += thread.intensity();
    }

//    m_hashrate = new Hashrate(threads.size(), controller);

    uv_mutex_init(&m_mutex);
    uv_rwlock_init(&m_rwlock);

//    m_timer = new uv_timer_t;
//    uv_timer_init(uv_default_loop(), m_timer);
//    uv_timer_start(m_timer, Workers::onTick, 500, 500);

//    size_t index = 0;
//    for (const CpuThread &thread : threads) {
//        Thread<CpuLaunchData> *handle = new Thread<CpuLaunchData>(index++, CpuLaunchData(m_status.algo, controller->config()->cpu(), thread));

//        m_workers.push_back(handle);
//        handle->start(WorkersLegacy::onReady);
//    }
}


//void Workers::stop()
//{
//    xmrig::Handle::close(m_timer);
//    m_hashrate->stop();

//    xmrig::Nonce::stop();

//    for (size_t i = 0; i < m_workers.size(); ++i) {
//        m_workers[i]->join();
//    }
//}


//#ifdef XMRIG_FEATURE_API
//void WorkersLegacy::threadsSummary(rapidjson::Document &doc)
//{
//    uv_mutex_lock(&m_mutex);
//    const uint64_t pages[2] = { m_status.hugePages, m_status.pages };
//    const uint64_t memory   = m_status.ways * xmrig::CnAlgo<>::memory(m_status.algo);
//    uv_mutex_unlock(&m_mutex);

//    auto &allocator = doc.GetAllocator();

//    rapidjson::Value hugepages(rapidjson::kArrayType);
//    hugepages.PushBack(pages[0], allocator);
//    hugepages.PushBack(pages[1], allocator);

//    doc.AddMember("hugepages", hugepages, allocator);
//    doc.AddMember("memory", memory, allocator);
//}
//#endif


//void WorkersLegacy::onTick(uv_timer_t *)
//{
//    using namespace xmrig;

//    for (Thread<CpuLaunchData> *handle : m_workers) {
//        if (!handle->worker()) {
//            return;
//        }

//        m_hashrate->add(handle->index(), handle->worker()->hashCount(), handle->worker()->timestamp());
//    }

//    if ((m_ticks++ & 0xF) == 0)  {
//        m_hashrate->updateHighest();
//    }
//}


void WorkersLegacy::start(xmrig::IWorker *worker)
{
//    const Worker *w = static_cast<const Worker *>(worker);

    uv_mutex_lock(&m_mutex);
    m_status.started++;
//    m_status.pages     += w->memory().pages;
//    m_status.hugePages += w->memory().hugePages;

    if (m_status.started == m_status.threads) {
        const double percent = (double) m_status.hugePages / m_status.pages * 100.0;
        const size_t memory  = m_status.ways * xmrig::CnAlgo<>::memory(m_status.algo) / 1024;

#       ifdef XMRIG_ALGO_RANDOMX
        if (m_status.algo.family() == xmrig::Algorithm::RANDOM_X) {
            LOG_INFO(GREEN_BOLD("READY (CPU)") " threads " CYAN_BOLD("%zu(%zu)") " memory " CYAN_BOLD("%zu KB") "",
                     m_status.threads, m_status.ways, memory);
        } else
#       endif
        {
            LOG_INFO(GREEN_BOLD("READY (CPU)") " threads " CYAN_BOLD("%zu(%zu)") " huge pages %s%zu/%zu %1.0f%%\x1B[0m memory " CYAN_BOLD("%zu KB") "",
                     m_status.threads, m_status.ways,
                     (m_status.hugePages == m_status.pages ? GREEN_BOLD_S : (m_status.hugePages == 0 ? RED_BOLD_S : YELLOW_BOLD_S)),
                     m_status.hugePages, m_status.pages, percent, memory);
        }
    }

    uv_mutex_unlock(&m_mutex);

    worker->start();
}
