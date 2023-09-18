/* XMRig
 * Copyright (c) 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include "backend/common/interfaces/IRxListener.h"
#include "base/io/Async.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/tools/Cvt.h"
#include "crypto/rx/RxBasicStorage.h"


#ifdef XMRIG_FEATURE_HWLOC
#   include "crypto/rx/RxNUMAStorage.h"
#endif


xmrig::RxQueue::RxQueue(IRxListener *listener) :
    m_listener(listener)
{
    m_async  = std::make_shared<Async>(this);
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


template<typename T>
bool xmrig::RxQueue::isReady(const T &seed)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    return isReadyUnsafe(seed);
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


template<typename T>
bool xmrig::RxQueue::isReadyUnsafe(const T &seed) const
{
    return m_storage != nullptr && m_storage->isAllocated() && m_state == STATE_IDLE && m_seed == seed;
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
                 Tags::randomx(),
                 item.nodeset.size() > 1 ? "s" : "",
                 item.seed.algorithm().name(),
                 item.threads,
                 Cvt::toHex(item.seed.data().data(), 8).data()
                 );

        m_storage->init(item.seed, item.threads, item.hugePages, item.oneGbPages, item.mode, item.priority);

        lock.lock();

        if (m_state == STATE_SHUTDOWN || !m_queue.empty()) {
            continue;
        }

        // Update seed here again in case there was more than one item in the queue
        m_seed = item.seed;
        m_state = STATE_IDLE;
        m_async->send();
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


namespace xmrig {


template bool RxQueue::isReady(const Job &);
template bool RxQueue::isReady(const RxSeed &);


} // namespace xmrig
