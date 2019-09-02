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
#include "api/ApiRouter.h"
#include "common/log/Log.h"
#include "core/Config.h"
#include "core/Controller.h"
#include "crypto/Argon2_constants.h"
#include "interfaces/IJobResultListener.h"
#include "rapidjson/document.h"
#include "workers/Handle.h"
#include "workers/Hashrate.h"
#include "workers/Workers.h"
#include "workers/Worker.h"


bool Workers::m_active = false;
bool Workers::m_enabled = true;
Hashrate *Workers::m_hashrate = nullptr;
xmrig::IJobResultListener *Workers::m_listener = nullptr;
xmrig::Job Workers::m_job;
Workers::LaunchStatus Workers::m_status;
std::atomic<int> Workers::m_paused;
std::atomic<uint64_t> Workers::m_sequence;
std::list<xmrig::JobResult> Workers::m_queue;
std::vector<Handle*> Workers::m_workers;
uint64_t Workers::m_ticks = 0;
uv_async_t Workers::m_async;
uv_mutex_t Workers::m_mutex;
uv_rwlock_t Workers::m_rwlock;
uv_timer_t Workers::m_timer;
xmrig::Controller *Workers::m_controller = nullptr;
std::atomic<int> Workers::m_totalThreads;


xmrig::Job Workers::job()
{
    uv_rwlock_rdlock(&m_rwlock);
    xmrig::Job job = m_job;
    uv_rwlock_rdunlock(&m_rwlock);

    return job;
}


void Workers::printHashrate(bool detail)
{
    assert(m_controller != nullptr);
    if (!m_controller) {
        return;
    }

    if (detail) {
        const bool isColors = m_controller->config()->isColors();
        char num1[8] = { 0 };
        char num2[8] = { 0 };
        char num3[8] = { 0 };

        Log::i()->text("%s|  TYPE   |   ID  | 10s H/s | 60s H/s | 15m H/s |", isColors ? "\x1B[1;37m" : "");

        size_t i = 0;
        for (const Handle *worker : m_workers) {
            for(int i = 0; i < worker->hasher()->deviceCount(); i++) {
                Log::i()->text("| %7s | %s%-2d | %7s | %7s | %7s |",
                               worker->hasher()->subType().c_str(),
                               worker->hasher()->subType(true).c_str(),
                               i,
                               Hashrate::format(m_hashrate->calc(worker->hasherId(), i, Hashrate::ShortInterval), num1,
                                                sizeof num1),
                               Hashrate::format(m_hashrate->calc(worker->hasherId(), i, Hashrate::MediumInterval), num2,
                                                sizeof num2),
                               Hashrate::format(m_hashrate->calc(worker->hasherId(), i, Hashrate::LargeInterval), num3,
                                                sizeof num3)
                );
            }
        }
    }

    m_hashrate->print();
}


void Workers::setEnabled(bool enabled)
{
    if (m_enabled == enabled) {
        return;
    }

    m_enabled = enabled;
    if (!m_active) {
        return;
    }

    m_paused = enabled ? 0 : 1;
    m_sequence++;
}


void Workers::setJob(const xmrig::Job &job, bool donate)
{
    uv_rwlock_wrlock(&m_rwlock);
    m_job = job;

    if (donate) {
        m_job.setPoolId(-1);
    }
    uv_rwlock_wrunlock(&m_rwlock);

    m_active = true;
    if (!m_enabled) {
        return;
    }

    m_sequence++;
    m_paused = 0;
}


bool Workers::start(xmrig::Controller *controller)
{
    m_controller = controller;

    const std::vector<xmrig::HasherConfig *> &hashers = controller->config()->hasherConfigs();
    m_status.algo    = controller->config()->algorithm().algo();
    m_status.variant = controller->config()->algorithm().variant();
    m_status.colors  = controller->config()->isColors();
    m_status.hashers = hashers.size();

    uv_mutex_init(&m_mutex);
    uv_rwlock_init(&m_rwlock);

    m_sequence = 1;
    m_paused   = 1;
    m_totalThreads = 0;

    uv_async_init(uv_default_loop(), &m_async, Workers::onResult);
    uv_timer_init(uv_default_loop(), &m_timer);
    uv_timer_start(&m_timer, Workers::onTick, 500, 500);

    uint32_t offset = 0;

    for (xmrig::HasherConfig *hasherConfig : hashers) {
        Handle *handle = new Handle((int)(m_workers.size()), controller->config(), hasherConfig, offset);
        if(handle->hasher() != nullptr) {
            offset += handle->computingThreads();
            m_totalThreads += handle->computingThreads();

            m_workers.push_back(handle);
            handle->start(Workers::onReady);
        }
    }

    if(m_workers.size() > 0) {
        Log::i()->text(m_status.colors ? GREEN_BOLD(" * Hashers initialization complete * ") : " * Hashers initialization complete * ");

        m_hashrate = new Hashrate(m_workers, controller);

        controller->save();
    }
    else {
        return false;
    }

    return true;
}


void Workers::stop()
{
    uv_timer_stop(&m_timer);
    m_hashrate->stop();

    uv_close(reinterpret_cast<uv_handle_t*>(&m_async), nullptr);
    m_paused   = 0;
    m_sequence = 0;

    for (size_t i = 0; i < m_workers.size(); ++i) {
        m_workers[i]->join();
    }
}


void Workers::submit(const xmrig::JobResult &result)
{
    uv_mutex_lock(&m_mutex);
    m_queue.push_back(result);
    uv_mutex_unlock(&m_mutex);

    uv_async_send(&m_async);
}


#ifndef XMRIG_NO_API
void Workers::hashersSummary(rapidjson::Document &doc)
{
    auto &allocator = doc.GetAllocator();

    rapidjson::Value hashers(rapidjson::kArrayType);

    for(int i = 0; i < m_workers.size(); i++) {
        Handle *worker = m_workers[i];
        for(int j=0; j < worker->hasher()->deviceCount(); j++) {
            rapidjson::Value hasherDoc(rapidjson::kObjectType);
            int multiplier = worker->hasher()->computingThreads() / worker->hasher()->deviceCount();

            xmrig::String type = worker->hasher()->type().data();
            xmrig::String id = (worker->hasher()->subType(true) + to_string(j)).data();
            DeviceInfo &deviceInfo = worker->hasher()->device(j * multiplier);
            xmrig::String device = deviceInfo.name.data();
            xmrig::String busId = deviceInfo.bus_id.data();

            hasherDoc.AddMember("type",  type.toJSON(doc), allocator);
            hasherDoc.AddMember("id",   id.toJSON(doc), allocator);
            hasherDoc.AddMember("device",   device.toJSON(doc), allocator);
            hasherDoc.AddMember("bus_id", busId.toJSON(doc), allocator);

            rapidjson::Value hashrateEntry(rapidjson::kArrayType);
            hashrateEntry.PushBack(ApiRouter::normalize(m_hashrate->calc(i, j, Hashrate::ShortInterval)), allocator);
            hashrateEntry.PushBack(ApiRouter::normalize(m_hashrate->calc(i, j, Hashrate::MediumInterval)), allocator);
            hashrateEntry.PushBack(ApiRouter::normalize(m_hashrate->calc(i, j, Hashrate::LargeInterval)), allocator);

            hasherDoc.AddMember("hashrate",   hashrateEntry, allocator);

            hashers.PushBack(hasherDoc, allocator);
        }
    }

    doc.AddMember("hashers", hashers, allocator);
}
#endif


void Workers::onReady(void *arg)
{
    auto handleArg = static_cast<Handle::HandleArg*>(arg);

    IWorker *worker = new Worker(handleArg->handle, handleArg->workerId);

    handleArg->handle->addWorker(worker);

    if (!worker->selfTest()) {
        LOG_ERR("hasher %zu error: \"hash self-test failed\".", worker->id());

        return;
    }

    start(worker);
}


void Workers::onResult(uv_async_t *handle)
{
    std::list<xmrig::JobResult> results;

    uv_mutex_lock(&m_mutex);
    while (!m_queue.empty()) {
        results.push_back(std::move(m_queue.front()));
        m_queue.pop_front();
    }
    uv_mutex_unlock(&m_mutex);

    for (auto result : results) {
        m_listener->onJobResult(result);
    }

    results.clear();
}


void Workers::onTick(uv_timer_t *handle)
{
    for (int h =0; h < m_workers.size(); h++) {
        Handle *handle = m_workers[h];

        std::vector<IWorker *> internalWorkers = handle->workers();
        if (internalWorkers.size() == 0)
            return;

        int deviceCount = handle->hasher()->deviceCount();
        int computingThreads = internalWorkers.size();
        int multiplier = computingThreads / deviceCount;

        for(int i = 0; i < deviceCount; i++) {
            uint64_t hashCount = 0;
            uint64_t timeStamp = 0;

            for(int j = 0; j < multiplier; j++) {
                hashCount += internalWorkers[i * multiplier + j]->hashCount();
                timeStamp = max(timeStamp, internalWorkers[i * multiplier + j]->timestamp());
            }

            m_hashrate->add(h, i, hashCount, timeStamp);
        }
    }

    if ((m_ticks++ & 0xF) == 0)  {
        m_hashrate->updateHighest();
    }
}


void Workers::start(IWorker *worker)
{
    const Worker *w = static_cast<const Worker *>(worker);

    uv_mutex_lock(&m_mutex);
    m_status.started++;

    if (m_status.started == m_status.hashers) {
/// TODO better status description
/*        if (m_status.colors) {
            LOG_INFO(GREEN_BOLD("READY (CPU)") " threads " CYAN_BOLD("%zu") " huge pages %s%zu/%zu %1.0f%%\x1B[0m memory " CYAN_BOLD("%.2f KB") "",
                     m_status.hashers,
                     (m_status.hugePages == m_status.pages ? "\x1B[1;32m" : (m_status.hugePages == 0 ? "\x1B[1;31m" : "\x1B[1;33m")),
                     m_status.hugePages, m_status.pages, percent, memory);
        }
        else {
            LOG_INFO("READY (CPU) threads %zu huge pages %zu/%zu %1.0f%% memory %zu KB",
                     m_status.hashers, m_status.hugePages, m_status.pages, percent, memory);
        } */
    }

    uv_mutex_unlock(&m_mutex);

    worker->start();
}
