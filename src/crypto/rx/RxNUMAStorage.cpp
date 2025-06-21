/* XMRig
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
 * Copyright (c) 2018-2023 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2023 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "crypto/rx/RxNUMAStorage.h"
#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/kernel/Platform.h"
#include "base/tools/Chrono.h"
#include "crypto/rx/RxAlgo.h"
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxDataset.h"
#include "crypto/rx/RxSeed.h"


#include <map>
#include <mutex>
#include <hwloc.h>
#include <thread>


namespace xmrig {


constexpr size_t oneMiB = 1024 * 1024;
static std::mutex mutex;


static int8_t bindToNUMANode(uint32_t nodeId)
{
    auto node = hwloc_get_numanode_obj_by_os_index(Cpu::info()->topology(), nodeId);
    if (!node) {
        return MEMBIND_FAIL_NODE;
    }

    if (Cpu::info()->membind(node->nodeset) > 0) {
        Platform::setThreadAffinity(static_cast<uint64_t>(hwloc_bitmap_first(node->cpuset)));

        return MEMBIND_SUCCESS;
    }

    return MEMBIND_FAIL_BIND;
}


static inline void printSkipped(uint32_t nodeId, const char *reason)
{
    LOG_WARN("%s" CYAN_BOLD("#%u ") RED_BOLD("skipped") YELLOW(" (%s)"), Tags::randomx(), nodeId, reason);
}


static inline void printDatasetReady(uint32_t nodeId, uint64_t ts)
{
    LOG_INFO("%s" CYAN_BOLD("#%u ") GREEN_BOLD("dataset ready") BLACK_BOLD(" (%" PRIu64 " ms)"), Tags::randomx(), nodeId, Chrono::steadyMSecs() - ts);
}


class RxNUMAStoragePrivate
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(RxNUMAStoragePrivate)

    inline explicit RxNUMAStoragePrivate(const std::vector<uint32_t> &nodeset) :
        m_nodeset(nodeset)
    {
        m_threads.reserve(nodeset.size());
    }


    inline ~RxNUMAStoragePrivate()
    {
        join();

        for (auto const &item : m_datasets) {
            delete item.second;
        }
    }

    inline bool isAllocated() const                     { return m_allocated; }
    inline bool isReady(const Job &job) const           { return m_ready && m_seed == job; }
    inline RxDataset *dataset(uint32_t nodeId) const    { return m_datasets.count(nodeId) ? m_datasets.at(nodeId) : m_datasets.at(m_nodeset.front()); }


    inline void setSeed(const RxSeed &seed)
    {
        m_ready = false;

        if (m_seed.algorithm() != seed.algorithm()) {
            RxAlgo::apply(seed.algorithm());
        }

        m_seed = seed;
    }


    inline bool createDatasets(bool hugePages, bool oneGbPages)
    {
        const uint64_t ts = Chrono::steadyMSecs();

        for (uint32_t node : m_nodeset) {
            m_threads.emplace_back(allocate, this, node, hugePages, oneGbPages);
        }

        join();

        if (isCacheRequired()) {
            std::thread thread(allocateCache, this, m_nodeset.front(), hugePages);
            thread.join();

            if (!m_cache) {
                return false;
            }
        }

        if (m_datasets.empty()) {
            m_datasets.insert({ m_nodeset.front(), new RxDataset(m_cache) });

            LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "failed to allocate RandomX datasets, switching to slow mode" BLACK_BOLD(" (%" PRIu64 " ms)"), Tags::randomx(), Chrono::steadyMSecs() - ts);
        }
        else {
            if (m_cache) {
                dataset(m_nodeset.front())->setCache(m_cache);
            }

            printAllocStatus(ts);
        }

        m_allocated = true;

        return true;
    }


    inline bool isCacheRequired() const
    {
        if (m_datasets.empty()) {
            return true;
        }

        for (const auto &kv : m_datasets) {
            if (kv.second->isOneGbPages()) {
                return false;
            }
        }

        return true;
    }


    inline void initDatasets(uint32_t threads, int priority)
    {
        uint64_t ts = Chrono::steadyMSecs();
        uint32_t id = 0;

        for (const auto &kv : m_datasets) {
            if (kv.second->cache()) {
                id = kv.first;
            }
        }

        auto primary = dataset(id);
        primary->init(m_seed.data(), threads, priority);

        printDatasetReady(id, ts);

        if (m_datasets.size() > 1) {
            for (auto const &item : m_datasets) {
                if (item.first == id) {
                    continue;
                }

                m_threads.emplace_back(copyDataset, item.second, item.first, primary->raw());
            }

            join();
        }

        m_ready = true;
    }


    inline HugePagesInfo hugePages() const
    {
        HugePagesInfo pages;
        for (auto const &item : m_datasets) {
            pages += item.second->hugePages();
        }

        return pages;
    }


private:
    static void allocate(RxNUMAStoragePrivate *d_ptr, uint32_t nodeId, bool hugePages, bool oneGbPages)
    {
        const uint64_t ts = Chrono::steadyMSecs();
        const int8_t   br = bindToNUMANode(nodeId);

        if (br < 0) {
            switch (br) {
            case MEMBIND_FAIL_SUPP:
                printSkipped(nodeId, "can't bind memory: hwloc_topology_get_support() failed");
                break;
            case MEMBIND_FAIL_NODE:
                printSkipped(nodeId, "can't bind memory: hwloc_get_numanode_obj_by_os_index() failed");
                break;
            case MEMBIND_FAIL_BIND:
                printSkipped(nodeId, "can't bind memory: Cpu::info()->membind() failed");
                break;
            }
            return;
        }

        auto dataset = new RxDataset(hugePages, oneGbPages, false, RxConfig::FastMode, nodeId);
        if (!dataset->get()) {
            printSkipped(nodeId, "failed to allocate dataset");

            delete dataset;
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);
        d_ptr->m_datasets.insert({ nodeId, dataset });
        RxNUMAStoragePrivate::printAllocStatus(dataset, nodeId, ts);
    }


    static void allocateCache(RxNUMAStoragePrivate *d_ptr, uint32_t nodeId, bool hugePages)
    {
        const uint64_t ts = Chrono::steadyMSecs();

        bindToNUMANode(nodeId);

        auto cache = new RxCache(hugePages, nodeId);
        if (!cache->get()) {
            delete cache;

            LOG_INFO("%s" RED_BOLD("failed to allocate RandomX memory") BLACK_BOLD(" (%" PRIu64 " ms)"), Tags::randomx(), Chrono::steadyMSecs() - ts);

            return;
        }

        std::lock_guard<std::mutex> lock(mutex);
        d_ptr->m_cache = cache;
        RxNUMAStoragePrivate::printAllocStatus(cache, nodeId, ts);
    }


    static void copyDataset(RxDataset *dst, uint32_t nodeId, const void *raw)
    {
        const uint64_t ts = Chrono::steadyMSecs();

        dst->setRaw(raw);

        printDatasetReady(nodeId, ts);
    }


    static void printAllocStatus(RxDataset *dataset, uint32_t nodeId, uint64_t ts)
    {
        const auto pages = dataset->hugePages();

        LOG_INFO("%s" CYAN_BOLD("#%u ") GREEN_BOLD("allocated") CYAN_BOLD(" %zu MB") " huge pages %s%3.0f%%" CLEAR BLACK_BOLD(" (%" PRIu64 " ms)"),
                 Tags::randomx(),
                 nodeId,
                 pages.size / oneMiB,
                 (pages.isFullyAllocated() ? GREEN_BOLD_S : RED_BOLD_S),
                 pages.percent(),
                 Chrono::steadyMSecs() - ts
                 );
    }


    static void printAllocStatus(RxCache *cache, uint32_t nodeId, uint64_t ts)
    {
        const auto pages = cache->hugePages();

        LOG_INFO("%s" CYAN_BOLD("#%u ") GREEN_BOLD("allocated") CYAN_BOLD(" %4zu MB") " huge pages %s%3.0f%%" CLEAR " %sJIT" BLACK_BOLD(" (%" PRIu64 " ms)"),
                 Tags::randomx(),
                 nodeId,
                 cache->size() / oneMiB,
                 (pages.isFullyAllocated() ? GREEN_BOLD_S : RED_BOLD_S),
                 pages.percent(),
                 cache->isJIT() ? GREEN_BOLD_S "+" : RED_BOLD_S "-",
                 Chrono::steadyMSecs() - ts
                 );
    }


    void printAllocStatus(uint64_t ts) const
    {
        auto pages = hugePages();

        LOG_INFO("%s" CYAN_BOLD("-- ") GREEN_BOLD("allocated") CYAN_BOLD(" %4zu MB") " huge pages %s%3.0f%% %u/%u" CLEAR BLACK_BOLD(" (%" PRIu64 " ms)"),
                 Tags::randomx(),
                 pages.size / oneMiB,
                 (pages.isFullyAllocated() ? GREEN_BOLD_S : (pages.allocated == 0 ? RED_BOLD_S : YELLOW_BOLD_S)),
                 pages.percent(),
                 pages.allocated,
                 pages.total,
                 Chrono::steadyMSecs() - ts
                 );
    }


    inline void join()
    {
        for (auto &thread : m_threads) {
            thread.join();
        }

        m_threads.clear();
    }


    bool m_allocated        = false;
    bool m_ready            = false;
    RxCache *m_cache        = nullptr;
    RxSeed m_seed;
    std::map<uint32_t, RxDataset *> m_datasets;
    std::vector<std::thread> m_threads;
    std::vector<uint32_t> m_nodeset;
};


} // namespace xmrig


xmrig::RxNUMAStorage::RxNUMAStorage(const std::vector<uint32_t> &nodeset) :
    d_ptr(new RxNUMAStoragePrivate(nodeset))
{
}


xmrig::RxNUMAStorage::~RxNUMAStorage()
{
    delete d_ptr;
}


bool xmrig::RxNUMAStorage::isAllocated() const
{
    return d_ptr->isAllocated();
}


xmrig::HugePagesInfo xmrig::RxNUMAStorage::hugePages() const
{
    if (!d_ptr->isAllocated()) {
        return {};
    }

    return d_ptr->hugePages();
}


xmrig::RxDataset *xmrig::RxNUMAStorage::dataset(const Job &job, uint32_t nodeId) const
{
    if (!d_ptr->isReady(job)) {
        return nullptr;
    }

    return d_ptr->dataset(nodeId);
}


void xmrig::RxNUMAStorage::init(const RxSeed &seed, uint32_t threads, bool hugePages, bool oneGbPages, RxConfig::Mode, int priority)
{
    d_ptr->setSeed(seed);

    if (!d_ptr->isAllocated() && !d_ptr->createDatasets(hugePages, oneGbPages)) {
        return;
    }

    d_ptr->initDatasets(threads, priority);
}
