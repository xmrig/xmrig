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


#include "backend/common/Hashrate.h"
#include "backend/common/Workers.h"
#include "backend/cpu/CpuBackend.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "base/tools/String.h"
#include "core/config/Config.h"
#include "core/Controller.h"


namespace xmrig {


extern template class Threads<CpuThread>;


class CpuBackendPrivate
{
public:
    inline CpuBackendPrivate(const Miner *miner, Controller *controller) :
        miner(miner),
        controller(controller)
    {
    }


    inline ~CpuBackendPrivate()
    {
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


    Algorithm algo;
    const Miner *miner;
    Controller *controller;
    CpuThreads threads;
    String profileName;
    Workers<CpuLaunchData> workers;
};


} // namespace xmrig


xmrig::CpuBackend::CpuBackend(const Miner *miner, Controller *controller) :
    d_ptr(new CpuBackendPrivate(miner, controller))
{

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

    const CpuConfig &cpu              = d_ptr->controller->config()->cpu();
    const Threads<CpuThread> &threads = cpu.threads();

    d_ptr->algo         = job.algorithm();
    d_ptr->profileName  = threads.profileName(job.algorithm());
    d_ptr->threads      = threads.get(d_ptr->profileName);

    LOG_INFO(GREEN_BOLD("CPU") " use profile " BLUE_BG(WHITE_BOLD_S " %s ") WHITE_BOLD_S " (" CYAN_BOLD("%zu") WHITE_BOLD(" threads)") " scratchpad " CYAN_BOLD("%zu KB"),
             d_ptr->profileName.data(),
             d_ptr->threads.size(),
             d_ptr->algo.memory() / 1024
             );

    d_ptr->workers.stop();

    for (const CpuThread &thread : d_ptr->threads) {
        d_ptr->workers.add(CpuLaunchData(d_ptr->miner, d_ptr->algo, cpu, thread));
    }

    d_ptr->workers.start();
}


void xmrig::CpuBackend::stop()
{
    d_ptr->workers.stop();
}


void xmrig::CpuBackend::tick(uint64_t ticks)
{
    d_ptr->workers.tick(ticks);
}
