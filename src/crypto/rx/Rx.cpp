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


#include "crypto/rx/Rx.h"

#include "backend/common/interfaces/IRxListener.h"
#include "backend/common/interfaces/IRxStorage.h"
#include "backend/common/Tags.h"
#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"
#include "base/kernel/Platform.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Buffer.h"
#include "base/tools/Chrono.h"
#include "base/tools/Handle.h"
#include "base/tools/Object.h"
#include "crypto/rx/RxAlgo.h"
#include "crypto/rx/RxBasicStorage.h"
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxConfig.h"
#include "crypto/rx/RxDataset.h"
#include "crypto/rx/RxSeed.h"


#include <atomic>
#include <map>
#include <mutex>
#include <thread>
#include <uv.h>


namespace xmrig {


class RxPrivate;


static const char *tag  = BLUE_BG(WHITE_BOLD_S " rx  ") " ";
static RxPrivate *d_ptr = nullptr;
static std::mutex mutex;


class RxPrivate
{
public:
    XMRIG_DISABLE_COPY_MOVE(RxPrivate)

    inline RxPrivate() :
        m_pending(0)
    {
        m_async = new uv_async_t;
        m_async->data = this;

        uv_async_init(uv_default_loop(), m_async, [](uv_async_t *handle) { static_cast<RxPrivate *>(handle->data)->onReady(); });
    }


    inline ~RxPrivate()
    {
        Handle::close(m_async);

        delete m_storage;
    }


    inline bool isReady(const Job &job) const                   { return pending() == 0 && m_seed == job; }
    inline RxDataset *dataset(const Job &job, uint32_t nodeId)  { return m_storage ? m_storage->dataset(job, nodeId) : nullptr; }
    inline std::pair<uint32_t, uint32_t> hugePages()            { return m_storage ? m_storage->hugePages() : std::pair<uint32_t, uint32_t>(0u, 0u); }
    inline uint64_t pending() const                             { return m_pending.load(std::memory_order_relaxed); }
    inline void asyncSend()                                     { --m_pending; if (pending() == 0) { uv_async_send(m_async); } }


    inline IRxStorage *storage()
    {
        if (!m_storage) {
            m_storage = new RxBasicStorage();
        }

        return m_storage;
    }


    static void initDataset(const RxSeed &seed, const std::vector<uint32_t> &nodeset, uint32_t threads, bool hugePages)
    {
        std::lock_guard<std::mutex> lock(mutex);

        LOG_INFO("%s" MAGENTA_BOLD("init dataset%s") " algo " WHITE_BOLD("%s (") CYAN_BOLD("%u") WHITE_BOLD(" threads)") BLACK_BOLD(" seed %s..."),
                 tag,
                 nodeset.size() > 1 ? "s" : "",
                 seed.algorithm().shortName(),
                 threads,
                 Buffer::toHex(seed.data().data(), 8).data()
                 );

        d_ptr->storage()->init(seed, threads, hugePages);
        d_ptr->asyncSend();
    }


    inline void setState(const Job &job, IRxListener *listener)
    {
        m_listener  = listener;
        m_seed      = job;

        ++m_pending;
    }


private:
    inline void onReady()
    {
        if (m_listener && pending() == 0) {
            m_listener->onDatasetReady();
        }
    }


    IRxListener *m_listener = nullptr;
    IRxStorage *m_storage   = nullptr;
    RxSeed m_seed;
    std::atomic<uint64_t> m_pending;
    uv_async_t *m_async     = nullptr;
};


} // namespace xmrig


const char *xmrig::rx_tag()
{
    return tag;
}


bool xmrig::Rx::init(const Job &job, const RxConfig &config, bool hugePages, IRxListener *listener)
{
    if (job.algorithm().family() != Algorithm::RANDOM_X) {
        return true;
    }

    if (d_ptr->isReady(job)) {
        return true;
    }

    d_ptr->setState(job, listener);

    std::thread thread(RxPrivate::initDataset, job, config.nodeset(), config.threads(), hugePages);
    thread.detach();

    return false;
}


bool xmrig::Rx::isReady(const Job &job)
{
    return d_ptr->isReady(job);
}


xmrig::RxDataset *xmrig::Rx::dataset(const Job &job, uint32_t nodeId)
{
    std::lock_guard<std::mutex> lock(mutex);

    return d_ptr->dataset(job, nodeId);
}


std::pair<uint32_t, uint32_t> xmrig::Rx::hugePages()
{
    std::lock_guard<std::mutex> lock(mutex);

    return d_ptr->hugePages();
}


void xmrig::Rx::destroy()
{
    delete d_ptr;

    d_ptr = nullptr;
}


void xmrig::Rx::init()
{
    d_ptr = new RxPrivate();
}
