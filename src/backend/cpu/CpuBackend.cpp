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


#include <uv.h>


#include "backend/common/Hashrate.h"
#include "backend/common/interfaces/IWorker.h"
#include "backend/common/Workers.h"
#include "backend/cpu/CpuBackend.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "base/tools/String.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "crypto/common/VirtualMemory.h"


namespace xmrig {


extern template class Threads<CpuThread>;


struct LaunchStatus
{
public:
    inline void reset()
    {
        hugePages = 0;
        memory    = 0;
        pages     = 0;
        started   = 0;
        threads   = 0;
        ways      = 0;
    }

    size_t hugePages;
    size_t memory;
    size_t pages;
    size_t started;
    size_t threads;
    size_t ways;
};


class CpuBackendPrivate
{
public:
    inline CpuBackendPrivate(Controller *controller) :
        controller(controller)
    {
        uv_mutex_init(&mutex);
    }


    inline ~CpuBackendPrivate()
    {
        uv_mutex_destroy(&mutex);
    }


    inline bool isReady(const Algorithm &nextAlgo) const
    {
        if (!algo.isValid()) {
            return false;
        }

        if (nextAlgo == algo) {
            return true;
        }

        const CpuThreads &nextThreads = controller->config()->cpu().threads().get(nextAlgo);

        return algo.memory() == nextAlgo.memory()
                && threads.size() == nextThreads.size()
                && std::equal(threads.begin(), threads.end(), nextThreads.begin());
    }


    inline void start(const Job &job)
    {
        const CpuConfig &cpu = controller->config()->cpu();

        algo         = job.algorithm();
        profileName  = cpu.threads().profileName(job.algorithm());
        threads      = cpu.threads().get(profileName);

        LOG_INFO(GREEN_BOLD("CPU") " use profile " BLUE_BG(WHITE_BOLD_S " %s ") WHITE_BOLD_S " (" CYAN_BOLD("%zu") WHITE_BOLD(" threads)") " scratchpad " CYAN_BOLD("%zu KB"),
                 profileName.data(),
                 threads.size(),
                 algo.memory() / 1024
                 );

        workers.stop();

        status.reset();
        status.memory   = algo.memory();
        status.threads  = threads.size();

        for (const CpuThread &thread : threads) {
            workers.add(CpuLaunchData(controller->miner(), algo, cpu, thread));

            status.ways += static_cast<size_t>(thread.intensity());
        }

        workers.start();
    }


    Algorithm algo;
    Controller *controller;
    CpuThreads threads;
    LaunchStatus status;
    String profileName;
    uv_mutex_t mutex;
    Workers<CpuLaunchData> workers;
};


} // namespace xmrig


xmrig::CpuBackend::CpuBackend(Controller *controller) :
    d_ptr(new CpuBackendPrivate(controller))
{
    d_ptr->workers.setBackend(this);
}


xmrig::CpuBackend::~CpuBackend()
{
    delete d_ptr;
}


const xmrig::Hashrate *xmrig::CpuBackend::hashrate() const
{
    return d_ptr->workers.hashrate();
}


const xmrig::String &xmrig::CpuBackend::profileName() const
{
    return d_ptr->profileName;
}


void xmrig::CpuBackend::printHashrate(bool details)
{
    if (!details || !hashrate()) {
        return;
    }

    char num[8 * 3] = { 0 };

    Log::print(WHITE_BOLD_S "|    CPU THREAD | AFFINITY | 10s H/s | 60s H/s | 15m H/s |");

    size_t i = 0;
    for (const CpuThread &thread : d_ptr->threads) {
         Log::print("| %13zu | %8" PRId64 " | %7s | %7s | %7s |",
                    i,
                    thread.affinity(),
                    Hashrate::format(hashrate()->calc(i, Hashrate::ShortInterval),  num,         sizeof num / 3),
                    Hashrate::format(hashrate()->calc(i, Hashrate::MediumInterval), num + 8,     sizeof num / 3),
                    Hashrate::format(hashrate()->calc(i, Hashrate::LargeInterval),  num + 8 * 2, sizeof num / 3)
                    );

         i++;
    }
}


void xmrig::CpuBackend::setJob(const Job &job)
{
    if (d_ptr->isReady(job.algorithm())) {
        return;
    }

    d_ptr->start(job);
}


void xmrig::CpuBackend::start(IWorker *worker)
{
    uv_mutex_lock(&d_ptr->mutex);

    const auto pages = worker->memory()->hugePages();

    d_ptr->status.started++;
    d_ptr->status.hugePages += pages.first;
    d_ptr->status.pages     += pages.second;

    if (d_ptr->status.started == d_ptr->status.threads) {
        const double percent = d_ptr->status.hugePages == 0 ? 0.0 : static_cast<double>(d_ptr->status.hugePages) / d_ptr->status.pages * 100.0;
        const size_t memory  = d_ptr->status.ways * d_ptr->status.memory / 1024;

        LOG_INFO(GREEN_BOLD("CPU READY") " threads " CYAN_BOLD("%zu(%zu)") " huge pages %s%zu/%zu %1.0f%%\x1B[0m memory " CYAN_BOLD("%zu KB") "",
                 d_ptr->status.threads, d_ptr->status.ways,
                 (d_ptr->status.hugePages == d_ptr->status.pages ? GREEN_BOLD_S : (d_ptr->status.hugePages == 0 ? RED_BOLD_S : YELLOW_BOLD_S)),
                 d_ptr->status.hugePages, d_ptr->status.pages, percent, memory);
    }

    uv_mutex_unlock(&d_ptr->mutex);

    worker->start();
}


void xmrig::CpuBackend::stop()
{
    d_ptr->workers.stop();
}


void xmrig::CpuBackend::tick(uint64_t ticks)
{
    d_ptr->workers.tick(ticks);
}
