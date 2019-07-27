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
#include <thread>
#include <uv.h>


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
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxDataset.h"


namespace xmrig {


static const char *tag  = BLUE_BG(WHITE_BOLD_S " rx ") " ";


class RxPrivate
{
public:
    inline RxPrivate()
    {
        uv_mutex_init(&mutex);
    }


    inline ~RxPrivate()
    {
        for (auto const &item : datasets) {
            delete item.second;
        }

        datasets.clear();

        uv_mutex_destroy(&mutex);
    }


    inline void lock()   { uv_mutex_lock(&mutex); }
    inline void unlock() { uv_mutex_unlock(&mutex); }


    static void allocate(RxPrivate *self, uint32_t nodeId)
    {
        const uint64_t ts = Chrono::steadyMSecs();

#       ifdef XMRIG_FEATURE_HWLOC
        if (self->numa) {
            hwloc_topology_t topology;
            hwloc_topology_init(&topology);
            hwloc_topology_load(topology);

            hwloc_obj_t node = hwloc_get_numanode_obj_by_os_index(topology, nodeId);
            if (node) {
                if (HwlocCpuInfo::has(HwlocCpuInfo::SET_THISTHREAD_MEMBIND)) {
                    hwloc_set_membind_nodeset(topology, node->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD);
                }

                Platform::setThreadAffinity(static_cast<uint64_t>(hwloc_bitmap_first(node->cpuset)));
            }

            hwloc_topology_destroy(topology);
        }
#       endif

        LOG_INFO("%s" CYAN_BOLD("#%u") MAGENTA_BOLD(" allocate") CYAN_BOLD(" %zu MB") BLACK_BOLD(" (%zu+%zu) for RandomX dataset & cache"),
                 tag,
                 nodeId,
                 (RxDataset::size() + RxCache::size()) / 1024 / 1024,
                 RxDataset::size() / 1024 / 1024,
                 RxCache::size() / 1024 / 1024
                 );

        RxDataset *dataset   = new RxDataset(self->hugePages);
        self->datasets[nodeId] = dataset;

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


    bool hugePages  = true;
    bool numa       = true;
    std::map<uint32_t, RxDataset *> datasets;
    uv_mutex_t mutex;
};


static RxPrivate *d_ptr = new RxPrivate();


} // namespace xmrig



bool xmrig::Rx::isReady(const Job &job, uint32_t nodeId)
{
    d_ptr->lock();
    const bool rc = isReady(job.seedHash(), job.algorithm(), d_ptr->numa ? nodeId : 0);
    d_ptr->unlock();

    return rc;
}



xmrig::RxDataset *xmrig::Rx::dataset(uint32_t nodeId)
{
    d_ptr->lock();
    RxDataset *dataset = d_ptr->datasets[d_ptr->numa ? nodeId : 0];
    d_ptr->unlock();

    return dataset;
}


void xmrig::Rx::init(const Job &job, int initThreads, bool hugePages, bool numa)
{
    if (job.algorithm().family() != Algorithm::RANDOM_X) {
        return;
    }

    d_ptr->lock();

    size_t ready = 0;

    for (auto const &item : d_ptr->datasets) {
        if (isReady(job.seedHash(), job.algorithm(), item.first)) {
            ready++;
        }
    }

    if (!d_ptr->datasets.empty() && ready == d_ptr->datasets.size()) {
        d_ptr->unlock();

        return;
    }

    d_ptr->hugePages       = hugePages;
    d_ptr->numa            = numa && Cpu::info()->nodes() > 1;
    const uint32_t threads = initThreads < 1 ? static_cast<uint32_t>(Cpu::info()->threads())
                                             : static_cast<uint32_t>(initThreads);

#   ifdef XMRIG_FEATURE_HWLOC
    if (d_ptr->numa) {
        for (uint32_t nodeId : HwlocCpuInfo::nodeIndexes()) {
            std::thread thread(initDataset, nodeId, job.seedHash(), job.algorithm(), threads);
            thread.detach();
        }
    }
    else
#   endif
    {
        std::thread thread(initDataset, 0, job.seedHash(), job.algorithm(), threads);
        thread.detach();
    }

    d_ptr->unlock();
}


void xmrig::Rx::stop()
{
    delete d_ptr;

    d_ptr = nullptr;
}


bool xmrig::Rx::isReady(const uint8_t *seed, const Algorithm &algorithm, uint32_t nodeId)
{
    return !d_ptr->datasets.empty() && d_ptr->datasets[nodeId] != nullptr && d_ptr->datasets[nodeId]->isReady(seed, algorithm);
}


void xmrig::Rx::initDataset(uint32_t nodeId, const uint8_t *seed, const Algorithm &algorithm, uint32_t threads)
{
    d_ptr->lock();

    RxDataset *dataset = d_ptr->datasets[nodeId];

    if (!dataset) {
#       ifdef XMRIG_FEATURE_HWLOC
        if (d_ptr->numa) {
            std::thread thread(RxPrivate::allocate, d_ptr, nodeId);
            thread.join();
        } else
#       endif
        {
            RxPrivate::allocate(d_ptr, nodeId);
        }

        dataset = d_ptr->datasets[nodeId];
    }

    if (!dataset->isReady(seed, algorithm)) {
        const uint64_t ts = Chrono::steadyMSecs();

        if (dataset->get() != nullptr) {
            LOG_INFO("%s" CYAN_BOLD("#%u") MAGENTA_BOLD(" init dataset") " algo " WHITE_BOLD("%s (") CYAN_BOLD("%u") WHITE_BOLD(" threads)") BLACK_BOLD(" seed %s..."),
                     tag,
                     nodeId,
                     algorithm.shortName(),
                     threads,
                     Buffer::toHex(seed, 8).data()
                     );
        }
        else {
            LOG_INFO("%s" CYAN_BOLD("#%u") MAGENTA_BOLD(" init cache") " algo " WHITE_BOLD("%s") BLACK_BOLD(" seed %s..."),
                     tag,
                     nodeId,
                     algorithm.shortName(),
                     Buffer::toHex(seed, 8).data()
                     );
        }

        dataset->init(seed, algorithm, threads);

        LOG_INFO("%s" CYAN_BOLD("#%u") GREEN(" init done") BLACK_BOLD(" (%" PRIu64 " ms)"), tag, nodeId, Chrono::steadyMSecs() - ts);
    }

    d_ptr->unlock();
}
