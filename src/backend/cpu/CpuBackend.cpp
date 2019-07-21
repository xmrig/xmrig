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
#include "backend/cpu/Cpu.h"
#include "backend/cpu/CpuBackend.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Chrono.h"
#include "base/tools/String.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "crypto/common/VirtualMemory.h"
#include "crypto/rx/Rx.h"
#include "crypto/rx/RxDataset.h"
#include "rapidjson/document.h"


namespace xmrig {


extern template class Threads<CpuThread>;


static const String kType = "cpu";


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
        ts        = Chrono::steadyMSecs();
    }

    size_t hugePages    = 0;
    size_t memory       = 0;
    size_t pages        = 0;
    size_t started      = 0;
    size_t threads      = 0;
    size_t ways         = 0;
    uint64_t ts         = 0;
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


    inline void start()
    {
        LOG_INFO(GREEN_BOLD("CPU") " use profile " BLUE_BG(WHITE_BOLD_S " %s ") WHITE_BOLD_S " (" CYAN_BOLD("%zu") WHITE_BOLD(" threads)") " scratchpad " CYAN_BOLD("%zu KB"),
                 profileName.data(),
                 threads.size(),
                 algo.memory() / 1024
                 );

        workers.stop();

        status.reset();
        status.memory   = algo.memory();
        status.threads  = threads.size();

        for (const CpuLaunchData &data : threads) {
            status.ways += static_cast<size_t>(data.intensity);
        }

        workers.start(threads);
    }


    Algorithm algo;
    Controller *controller;
    LaunchStatus status;
    std::vector<CpuLaunchData> threads;
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


bool xmrig::CpuBackend::isEnabled() const
{
    return d_ptr->controller->config()->cpu().isEnabled();
}


bool xmrig::CpuBackend::isEnabled(const Algorithm &algorithm) const
{
    return !d_ptr->controller->config()->cpu().threads().get(algorithm).empty();
}


const xmrig::Hashrate *xmrig::CpuBackend::hashrate() const
{
    return d_ptr->workers.hashrate();
}


const xmrig::String &xmrig::CpuBackend::profileName() const
{
    return d_ptr->profileName;
}


const xmrig::String &xmrig::CpuBackend::type() const
{
    return kType;
}


void xmrig::CpuBackend::printHashrate(bool details)
{
    if (!details || !hashrate()) {
        return;
    }

    char num[8 * 3] = { 0 };

    Log::print(WHITE_BOLD_S "|    CPU THREAD | AFFINITY | 10s H/s | 60s H/s | 15m H/s |");

    size_t i = 0;
    for (const CpuLaunchData &data : d_ptr->threads) {
         Log::print("| %13zu | %8" PRId64 " | %7s | %7s | %7s |",
                    i,
                    data.affinity,
                    Hashrate::format(hashrate()->calc(i, Hashrate::ShortInterval),  num,         sizeof num / 3),
                    Hashrate::format(hashrate()->calc(i, Hashrate::MediumInterval), num + 8,     sizeof num / 3),
                    Hashrate::format(hashrate()->calc(i, Hashrate::LargeInterval),  num + 8 * 2, sizeof num / 3)
                    );

         i++;
    }
}


void xmrig::CpuBackend::setJob(const Job &job)
{
    if (!isEnabled()) {
        d_ptr->workers.stop();
        d_ptr->threads.clear();
        return;
    }

    const CpuConfig &cpu = d_ptr->controller->config()->cpu();

    std::vector<CpuLaunchData> threads = cpu.get(d_ptr->controller->miner(), job.algorithm());
    if (d_ptr->threads.size() == threads.size() && std::equal(d_ptr->threads.begin(), d_ptr->threads.end(), threads.begin())) {
        return;
    }

    d_ptr->algo         = job.algorithm();
    d_ptr->profileName  = cpu.threads().profileName(job.algorithm());

    if (d_ptr->profileName.isNull() || threads.empty()) {
        d_ptr->workers.stop();

        LOG_WARN(YELLOW_BOLD_S "CPU disabled, no suitable configuration for algo %s", job.algorithm().shortName());

        return;
    }

    d_ptr->threads = std::move(threads);
    d_ptr->start();
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

        LOG_INFO(GREEN_BOLD("CPU READY") " threads " CYAN_BOLD("%zu(%zu)") " huge pages %s%zu/%zu %1.0f%%\x1B[0m memory " CYAN_BOLD("%zu KB") BLACK_BOLD(" (%" PRIu64 " ms)"),
                 d_ptr->status.threads, d_ptr->status.ways,
                 (d_ptr->status.hugePages == d_ptr->status.pages ? GREEN_BOLD_S : (d_ptr->status.hugePages == 0 ? RED_BOLD_S : YELLOW_BOLD_S)),
                 d_ptr->status.hugePages, d_ptr->status.pages, percent, memory,
                 Chrono::steadyMSecs() - d_ptr->status.ts
                 );
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


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::CpuBackend::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator         = doc.GetAllocator();
    const CpuConfig &cpu    = d_ptr->controller->config()->cpu();

    Value out(kObjectType);
    out.AddMember("type",       type().toJSON(), allocator);
    out.AddMember("enabled",    isEnabled(), allocator);
    out.AddMember("algo",       d_ptr->algo.toJSON(), allocator);
    out.AddMember("profile",    profileName().toJSON(), allocator);
    out.AddMember("hw-aes",     cpu.isHwAES(), allocator);
    out.AddMember("priority",   cpu.priority(), allocator);

#   ifdef XMRIG_FEATURE_ASM
    const Assembly assembly = Cpu::assembly(cpu.assembly());
    out.AddMember("asm", assembly.toJSON(), allocator);
#   else
    out.AddMember("asm", false, allocator);
#   endif

    uv_mutex_lock(&d_ptr->mutex);
    uint64_t pages[2] = { d_ptr->status.hugePages, d_ptr->status.pages };
    const size_t ways = d_ptr->status.ways;
    uv_mutex_unlock(&d_ptr->mutex);

#   ifdef XMRIG_ALGO_RANDOMX
    if (d_ptr->algo.family() == Algorithm::RANDOM_X) {
        RxDataset *dataset = Rx::dataset();
        if (dataset) {
            const auto rxPages = dataset->hugePages();
            pages[0] += rxPages.first;
            pages[1] += rxPages.second;
        }
    }
#   endif

    rapidjson::Value hugepages(rapidjson::kArrayType);
    hugepages.PushBack(pages[0], allocator);
    hugepages.PushBack(pages[1], allocator);

    out.AddMember("hugepages", hugepages, allocator);
    out.AddMember("memory",    static_cast<uint64_t>(d_ptr->algo.isValid() ? (ways * d_ptr->algo.memory()) : 0), allocator);

    if (d_ptr->threads.empty() || !hashrate()) {
        return out;
    }

    Value threads(kArrayType);
    const Hashrate *hr = hashrate();

    size_t i = 0;
    for (const CpuLaunchData &data : d_ptr->threads) {
        Value thread(kObjectType);
        thread.AddMember("intensity",   data.intensity, allocator);
        thread.AddMember("affinity",    data.affinity, allocator);
        thread.AddMember("av",          data.av(), allocator);

        Value hashrate(kArrayType);
        hashrate.PushBack(Hashrate::normalize(hr->calc(i, Hashrate::ShortInterval)),  allocator);
        hashrate.PushBack(Hashrate::normalize(hr->calc(i, Hashrate::MediumInterval)), allocator);
        hashrate.PushBack(Hashrate::normalize(hr->calc(i, Hashrate::LargeInterval)),  allocator);

        i++;

        thread.AddMember("hashrate", hashrate, allocator);
        threads.PushBack(thread, allocator);
    }

    out.AddMember("threads", threads, allocator);

    return out;
}
#endif
