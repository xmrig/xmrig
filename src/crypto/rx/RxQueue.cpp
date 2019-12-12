/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
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


#include "crypto/rx/RxQueue.h"
#include "backend/common/Tags.h"
#include "base/io/log/Log.h"
#include "crypto/rx/RxBasicStorage.h"
#include "base/tools/Handle.h"
#include "backend/common/interfaces/IRxListener.h"


#ifdef XMRIG_FEATURE_HWLOC
#   include "crypto/rx/RxNUMAStorage.h"
#endif


xmrig::RxQueue::RxQueue(IRxListener *listener) :
    m_listener(listener)
{
    m_async = new uv_async_t;
    m_async->data = this;

    uv_async_init(uv_default_loop(), m_async, [](uv_async_t *handle) { static_cast<RxQueue *>(handle->data)->onReady(); });

    m_thread = std::thread(&RxQueue::backgroundInit, this);
}


xmrig::RxQueue::~RxQueue()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_state = STATE_SHUTDOWN;
    lock.unlock();

    m_cv.notify_one();

    m_thread.join();

    delete m_storage;

    Handle::close(m_async);
}


bool xmrig::RxQueue::isReady(const Job &job)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    return isReadyUnsafe(job);
}


xmrig::RxDataset *xmrig::RxQueue::dataset(const Job &job, uint32_t nodeId)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (isReadyUnsafe(job)) {
        return m_storage->dataset(job, nodeId);
    }

    return nullptr;
}


xmrig::HugePagesInfo xmrig::RxQueue::hugePages()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    return m_storage && m_state == STATE_IDLE ? m_storage->hugePages() : HugePagesInfo();
}


void xmrig::RxQueue::enqueue(const RxSeed &seed, const std::vector<uint32_t> &nodeset, uint32_t threads, bool hugePages, bool oneGbPages, RxConfig::Mode mode, int priority)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    if (!m_storage) {
#       ifdef XMRIG_FEATURE_HWLOC
        if (!nodeset.empty()) {
            m_storage = new RxNUMAStorage(nodeset);
        }
        else
#       endif
        {
            m_storage = new RxBasicStorage();
        }
    }

    if (m_state == STATE_PENDING && m_seed == seed) {
        return;
    }

    m_queue.emplace_back(seed, nodeset, threads, hugePages, oneGbPages, mode, priority);
    m_seed  = seed;
    m_state = STATE_PENDING;

    lock.unlock();

    m_cv.notify_one();
}


bool xmrig::RxQueue::isReadyUnsafe(const Job &job) const
{
    return m_storage != nullptr && m_state == STATE_IDLE && m_seed == job;
}


void xmrig::RxQueue::backgroundInit()
{
    while (m_state != STATE_SHUTDOWN) {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (m_state == STATE_IDLE) {
            m_cv.wait(lock, [this]{ return m_state != STATE_IDLE; });
        }

        if (m_state != STATE_PENDING) {
            continue;
        }

        const auto item = m_queue.back();
        m_queue.clear();

        lock.unlock();

        LOG_INFO("%s" MAGENTA_BOLD("init dataset%s") " algo " WHITE_BOLD("%s (") CYAN_BOLD("%u") WHITE_BOLD(" threads)") BLACK_BOLD(" seed %s..."),
                 rx_tag(),
                 item.nodeset.size() > 1 ? "s" : "",
                 item.seed.algorithm().shortName(),
                 item.threads,
                 Buffer::toHex(item.seed.data().data(), 8).data()
                 );

        m_storage->init(item.seed, item.threads, item.hugePages, item.oneGbPages, item.mode, item.priority);

        lock = std::unique_lock<std::mutex>(m_mutex);

        if (m_state == STATE_SHUTDOWN || !m_queue.empty()) {
            continue;
        }

        m_state = STATE_IDLE;
        uv_async_send(m_async);
    }
}


void xmrig::RxQueue::onReady()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    const bool ready = m_listener && m_state == STATE_IDLE;
    lock.unlock();

    if (ready) {
        m_listener->onDatasetReady();
    }
}
