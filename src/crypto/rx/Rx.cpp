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


#include <thread>
#include <uv.h>


#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Buffer.h"
#include "base/tools/Chrono.h"
#include "crypto/rx/Rx.h"
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxDataset.h"


namespace xmrig {


class RxPrivate
{
public:
    inline RxPrivate()
    {
        uv_mutex_init(&mutex);
    }


    inline ~RxPrivate()
    {
        for (RxDataset *dataset : datasets) {
            delete dataset;
        }

        uv_mutex_destroy(&mutex);
    }


    inline void lock()   { uv_mutex_lock(&mutex); }
    inline void unlock() { uv_mutex_unlock(&mutex); }


    std::vector<RxDataset *> datasets;
    uv_mutex_t mutex;
};


static RxPrivate *d_ptr = new RxPrivate();
static const char *tag  = BLUE_BG(WHITE_BOLD_S " rx ");


} // namespace xmrig



bool xmrig::Rx::isReady(const Job &job, int64_t)
{
    d_ptr->lock();
    const bool rc = isReady(job.seedHash(), job.algorithm());
    d_ptr->unlock();

    return rc;
}



xmrig::RxDataset *xmrig::Rx::dataset(int64_t)
{
    d_ptr->lock();
    RxDataset *dataset = d_ptr->datasets[0];
    d_ptr->unlock();

    return dataset;
}


void xmrig::Rx::init(const Job &job, int initThreads, bool hugePages)
{
    d_ptr->lock();
    if (d_ptr->datasets.empty()) {
        d_ptr->datasets.push_back(nullptr);
    }

    if (isReady(job.seedHash(), job.algorithm())) {
        d_ptr->unlock();

        return;
    }

    const uint32_t threads  = initThreads < 1 ? static_cast<uint32_t>(Cpu::info()->threads())
                                              : static_cast<uint32_t>(initThreads);

    std::thread thread(initDataset, 0, job.seedHash(), job.algorithm(), threads, hugePages);
    thread.detach();

    d_ptr->unlock();
}


void xmrig::Rx::stop()
{
    delete d_ptr;

    d_ptr = nullptr;
}


bool xmrig::Rx::isReady(const uint8_t *seed, const Algorithm &algorithm)
{
    return !d_ptr->datasets.empty() && d_ptr->datasets[0] != nullptr && d_ptr->datasets[0]->isReady(seed, algorithm);
}


void xmrig::Rx::initDataset(size_t index, const uint8_t *seed, const Algorithm &algorithm, uint32_t threads, bool hugePages)
{
    d_ptr->lock();

    if (!d_ptr->datasets[index]) {
        const uint64_t ts = Chrono::steadyMSecs();

        LOG_INFO("%s" MAGENTA_BOLD(" allocate") CYAN_BOLD(" %zu MiB") BLACK_BOLD(" (%zu+%zu) for RandomX dataset & cache"),
                 tag,
                 (RxDataset::size() + RxCache::size()) / 1024 / 1024,
                 RxDataset::size() / 1024 / 1024,
                 RxCache::size() / 1024 / 1024
                 );

        d_ptr->datasets[index] = new RxDataset(hugePages);

        if (d_ptr->datasets[index]->get() != nullptr) {
            const auto hugePages = d_ptr->datasets[index]->hugePages();
            const double percent = hugePages.first == 0 ? 0.0 : static_cast<double>(hugePages.first) / hugePages.second * 100.0;

            LOG_INFO("%s" GREEN(" allocate done") " huge pages %s%u/%u %1.0f%%" CLEAR " %sJIT" BLACK_BOLD(" (%" PRIu64 " ms)"),
                     tag,
                     (hugePages.first == hugePages.second ? GREEN_BOLD_S : (hugePages.first == 0 ? RED_BOLD_S : YELLOW_BOLD_S)),
                     hugePages.first,
                     hugePages.second,
                     percent,
                     d_ptr->datasets[index]->cache()->isJIT() ? GREEN_BOLD_S "+" : RED_BOLD_S "-",
                     Chrono::steadyMSecs() - ts
                     );
        }
        else {
            LOG_WARN(CLEAR "%s" YELLOW_BOLD_S " failed to allocate RandomX dataset, switching to slow mode", tag);
        }
    }

    if (!d_ptr->datasets[index]->isReady(seed, algorithm)) {
        const uint64_t ts = Chrono::steadyMSecs();

        if (d_ptr->datasets[index]->get() != nullptr) {
            LOG_INFO("%s" MAGENTA_BOLD(" init dataset") " algo " WHITE_BOLD("%s (") CYAN_BOLD("%u") WHITE_BOLD(" threads)") BLACK_BOLD(" seed %s..."),
                     tag,
                     algorithm.shortName(),
                     threads,
                     Buffer::toHex(seed, 8).data()
                     );
        }
        else {
            LOG_INFO("%s" MAGENTA_BOLD(" init cache") " algo " WHITE_BOLD("%s") BLACK_BOLD(" seed %s..."),
                     tag,
                     algorithm.shortName(),
                     Buffer::toHex(seed, 8).data()
                     );
        }

        d_ptr->datasets[index]->init(seed, algorithm, threads);

        LOG_INFO("%s" GREEN(" init done") BLACK_BOLD(" (%" PRIu64 " ms)"), tag, Chrono::steadyMSecs() - ts);
    }

    d_ptr->unlock();
}
