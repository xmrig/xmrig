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


#include <map>
#include <mutex>
#include <thread>


#ifdef XMRIG_FEATURE_HWLOC
#   include <hwloc.h>
#   include "backend/cpu/platform/HwlocCpuInfo.h"
#endif


#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"
#include "base/kernel/Platform.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Buffer.h"
#include "base/tools/Chrono.h"
#include "crypto/rx/Rx.h"
#include "crypto/rx/RxAlgo.h"
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxDataset.h"


namespace xmrig {


class RxPrivate;


static const char *tag  = BLUE_BG(WHITE_BOLD_S " rx ") " ";
static RxPrivate *d_ptr = nullptr;


#ifdef XMRIG_FEATURE_HWLOC
static void bindToNUMANode(uint32_t nodeId)
{
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    hwloc_obj_t node = hwloc_get_numanode_obj_by_os_index(topology, nodeId);
    if (node) {
        if (HwlocCpuInfo::has(HwlocCpuInfo::SET_THISTHREAD_MEMBIND)) {
#           if HWLOC_API_VERSION >= 0x20000
            hwloc_set_membind(topology, node->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_BYNODESET);
#           else
            hwloc_set_membind_nodeset(topology, node->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD);
#           endif
        }

        Platform::setThreadAffinity(static_cast<uint64_t>(hwloc_bitmap_first(node->cpuset)));
    }

    hwloc_topology_destroy(topology);
}
#else
inline static void bindToNUMANode(uint32_t) {}
#endif


class RxPrivate
{
public:
    inline RxPrivate() :
        m_seed()
    {
#       ifdef XMRIG_FEATURE_HWLOC
        if (Cpu::info()->nodes() > 1) {
            for (uint32_t nodeId : HwlocCpuInfo::nodeIndexes()) {
                datasets.insert({ nodeId, nullptr });
            }
        }
        else
#       endif
        {
            datasets.insert({ 0, nullptr });
        }
    }


    inline ~RxPrivate()
    {
        for (auto const &item : datasets) {
            delete item.second;
        }

        datasets.clear();
    }


    inline bool isNUMA() const                  { return m_numa; }
    inline const Algorithm &algorithm() const   { return m_algorithm; }
    inline const uint8_t *seed() const          { return m_seed; }
    inline size_t count() const                 { return isNUMA() ? datasets.size() : 1; }


    static void allocate(uint32_t nodeId)
    {
        const uint64_t ts = Chrono::steadyMSecs();

        if (d_ptr->isNUMA()) {
            bindToNUMANode(nodeId);
        }

        LOG_INFO("%s" CYAN_BOLD("#%u") MAGENTA_BOLD(" allocate") CYAN_BOLD(" %zu MB") BLACK_BOLD(" (%zu+%zu) for RandomX dataset & cache"),
                 tag,
                 nodeId,
                 (RxDataset::size() + RxCache::size()) / 1024 / 1024,
                 RxDataset::size() / 1024 / 1024,
                 RxCache::size() / 1024 / 1024
                 );

        RxDataset *dataset      = new RxDataset(d_ptr->m_hugePages);
        d_ptr->datasets[nodeId] = dataset;

        if (dataset->get() != nullptr) {
            const auto hugePages = dataset->hugePages();
            const double percent = hugePages.first == 0 ? 0.0 : static_cast<double>(hugePages.first) / hugePages.second * 100.0;

            LOG_INFO("%s" CYAN_BOLD("#%u") GREEN(" allocate done") " huge pages %s%u/%u %1.0f%%" CLEAR " %sJIT" BLACK_BOLD(" (%" PRIu64 " ms)"),
                     tag,
                     nodeId,
                     (hugePages.first == hugePages.second ? GREEN_BOLD_S : (hugePages.first == 0 ? RED_BOLD_S : YELLOW_BOLD_S)),
                     hugePages.first,
                     hugePages.second,
                     percent,
                     dataset->cache()->isJIT() ? GREEN_BOLD_S "+" : RED_BOLD_S "-",
                     Chrono::steadyMSecs() - ts
                     );
        }
        else {
            LOG_WARN(CLEAR "%s" CYAN_BOLD("#%u") YELLOW_BOLD_S " failed to allocate RandomX dataset, switching to slow mode", tag, nodeId);
        }
    }


    static void initDataset(uint32_t nodeId, uint32_t threads)
    {
        std::lock_guard<std::mutex> lock(d_ptr->mutex);

        const uint64_t ts = Chrono::steadyMSecs();

        d_ptr->getOrAllocate(nodeId)->init(d_ptr->seed(), threads);
        d_ptr->m_ready++;

        LOG_INFO("%s" CYAN_BOLD("#%u") GREEN(" init done") BLACK_BOLD(" (%" PRIu64 " ms)"), tag, nodeId, Chrono::steadyMSecs() - ts);
    }


    inline RxDataset *getOrAllocate(uint32_t nodeId)
    {
        RxDataset *dataset = datasets.at(nodeId);

        if (dataset == nullptr) {
    #       ifdef XMRIG_FEATURE_HWLOC
            if (d_ptr->isNUMA()) {
                std::thread thread(allocate, nodeId);
                thread.join();
            } else
    #       endif
            {
                allocate(nodeId);
            }

            dataset = datasets.at(nodeId);
        }

        return dataset;
    }


    inline void setState(const Job &job, bool hugePages, bool numa)
    {
        if (m_algorithm != job.algorithm()) {
            m_algorithm = RxAlgo::apply(job.algorithm());
        }

        m_ready     = 0;
        m_numa      = numa && Cpu::info()->nodes() > 1;
        m_hugePages = hugePages;

        memcpy(m_seed, job.seedHash(), sizeof(m_seed));
    }


    inline bool isReady(const Job &job)
    {
        return m_ready == count() && m_algorithm == job.algorithm() && memcmp(m_seed, job.seedHash(), sizeof(m_seed)) == 0;
    }


    std::map<uint32_t, RxDataset *> datasets;
    std::mutex mutex;

private:
    bool m_hugePages  = true;
    bool m_numa       = true;
    Algorithm m_algorithm;
    size_t m_ready = 0;
    uint8_t m_seed[32];
};


} // namespace xmrig


xmrig::RxDataset *xmrig::Rx::dataset(const Job &job, uint32_t nodeId)
{
    std::lock_guard<std::mutex> lock(d_ptr->mutex);
    if (!d_ptr->isReady(job)) {
        return nullptr;
    }

    return d_ptr->datasets.at(d_ptr->isNUMA() ? nodeId : 0);
}


std::pair<size_t, size_t> xmrig::Rx::hugePages()
{
    std::pair<size_t, size_t> pages(0, 0);
    std::lock_guard<std::mutex> lock(d_ptr->mutex);

    for (auto const &item : d_ptr->datasets) {
        if (!item.second) {
            continue;
        }

        const auto p = item.second->hugePages();
        pages.first  += p.first;
        pages.second += p.second;
    }

    return pages;
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


void xmrig::Rx::init(const Job &job, int initThreads, bool hugePages, bool numa)
{
    if (job.algorithm().family() != Algorithm::RANDOM_X) {
        return;
    }

    std::lock_guard<std::mutex> lock(d_ptr->mutex);

    if (d_ptr->isReady(job)) {
        return;
    }

    d_ptr->setState(job, hugePages, numa);
    const uint32_t threads = initThreads < 1 ? static_cast<uint32_t>(Cpu::info()->threads()) : static_cast<uint32_t>(initThreads);
    const String buf       = Buffer::toHex(job.seedHash(), 8);

    LOG_INFO("%s" MAGENTA_BOLD("init dataset%s") " algo " WHITE_BOLD("%s (") CYAN_BOLD("%u") WHITE_BOLD(" threads)") BLACK_BOLD(" seed %s..."),
             tag,
             d_ptr->count() > 1 ? "s" : "",
             job.algorithm().shortName(),
             threads,
             buf.data()
             );

#   ifdef XMRIG_FEATURE_HWLOC
    if (d_ptr->isNUMA()) {
        for (auto const &item : d_ptr->datasets) {
            std::thread thread(RxPrivate::initDataset, item.first, threads);
            thread.detach();
        }
    }
    else
#   endif
    {
        std::thread thread(RxPrivate::initDataset, 0, threads);
        thread.detach();
    }
}
