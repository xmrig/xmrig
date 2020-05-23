/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <mutex>
#include <string>


#include "backend/cuda/CudaBackend.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/common/Hashrate.h"
#include "backend/common/interfaces/IWorker.h"
#include "backend/common/Tags.h"
#include "backend/common/Workers.h"
#include "backend/cuda/CudaConfig.h"
#include "backend/cuda/CudaThreads.h"
#include "backend/cuda/CudaWorker.h"
#include "backend/cuda/wrappers/CudaDevice.h"
#include "backend/cuda/wrappers/CudaLib.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Chrono.h"
#include "base/tools/String.h"
#include "core/config/Config.h"
#include "core/Controller.h"


#ifdef XMRIG_ALGO_ASTROBWT
#   include "backend/cuda/runners/CudaAstroBWTRunner.h"
#endif


#ifdef XMRIG_FEATURE_API
#   include "base/api/interfaces/IApiRequest.h"
#endif


#ifdef XMRIG_FEATURE_NVML
#include "backend/cuda/wrappers/NvmlLib.h"

namespace xmrig { static const char *kNvmlLabel = "NVML"; }
#endif


namespace xmrig {


extern template class Threads<CudaThreads>;


constexpr const size_t oneMiB   = 1024U * 1024U;
static const char *kLabel       = "CUDA";
static const char *tag          = GREEN_BG_BOLD(WHITE_BOLD_S " nv  ");
static const String kType       = "cuda";
static std::mutex mutex;



static void printDisabled(const char *label, const char *reason)
{
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") RED_BOLD("disabled") "%s", label, reason);
}


struct CudaLaunchStatus
{
public:
    inline size_t threads() const { return m_threads; }

    inline bool started(bool ready)
    {
        ready ? m_started++ : m_errors++;

        return (m_started + m_errors) == m_threads;
    }

    inline void start(size_t threads)
    {
        m_started         = 0;
        m_errors          = 0;
        m_threads         = threads;
        m_ts              = Chrono::steadyMSecs();
        CudaWorker::ready = false;
    }

    inline void print() const
    {
        if (m_started == 0) {
            LOG_ERR("%s " RED_BOLD("disabled") YELLOW(" (failed to start threads)"), tag);

            return;
        }

        LOG_INFO("%s" GREEN_BOLD(" READY") " threads " "%s%zu/%zu" BLACK_BOLD(" (%" PRIu64 " ms)"),
                 tag,
                 m_errors == 0 ? CYAN_BOLD_S : YELLOW_BOLD_S,
                 m_started,
                 m_threads,
                 Chrono::steadyMSecs() - m_ts
                 );
    }

private:
    size_t m_errors     = 0;
    size_t m_started    = 0;
    size_t m_threads    = 0;
    uint64_t m_ts       = 0;
};


class CudaBackendPrivate
{
public:
    inline CudaBackendPrivate(Controller *controller) :
        controller(controller)
    {
        init(controller->config()->cuda());
    }


    void init(const CudaConfig &cuda)
    {
        if (!cuda.isEnabled()) {
            return printDisabled(kLabel, "");
        }

        if (!CudaLib::init(cuda.loader())) {
            return printDisabled(kLabel, RED_S " (failed to load CUDA plugin)");
        }

        runtimeVersion = CudaLib::runtimeVersion();
        driverVersion  = CudaLib::driverVersion();

        if (!runtimeVersion || !driverVersion || !CudaLib::deviceCount()) {
            return printDisabled(kLabel, RED_S " (no devices)");
        }

        if (!devices.empty()) {
            return;
        }

        devices = CudaLib::devices(cuda.bfactor(), cuda.bsleep(), cuda.devicesHint());
        if (devices.empty()) {
            return printDisabled(kLabel, RED_S " (no devices)");
        }

        Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") WHITE_BOLD("%s") "/" WHITE_BOLD("%s") BLACK_BOLD("/%s"), kLabel,
                   CudaLib::version(runtimeVersion).c_str(), CudaLib::version(driverVersion).c_str(), CudaLib::pluginVersion());

#       ifdef XMRIG_FEATURE_NVML
        if (cuda.isNvmlEnabled()) {
            if (NvmlLib::init(cuda.nvmlLoader())) {
                NvmlLib::assign(devices);

                Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") WHITE_BOLD("%s") "/" GREEN_BOLD("%s") " press " MAGENTA_BG(WHITE_BOLD_S "e") " for health report",
                           kNvmlLabel,
                           NvmlLib::version(),
                           NvmlLib::driverVersion()
                           );
            }
            else {
                printDisabled(kNvmlLabel, RED_S " (failed to load NVML)");
            }
        }
        else {
            printDisabled(kNvmlLabel, "");
        }
#       endif

        for (const CudaDevice &device : devices) {
            Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("#%zu") YELLOW(" %s") GREEN_BOLD(" %s ") WHITE_BOLD("%u/%u MHz") " smx:" WHITE_BOLD("%u") " arch:" WHITE_BOLD("%u%u") " mem:" CYAN("%zu/%zu") " MB",
                       "CUDA GPU",
                       device.index(),
                       device.topology().toString().data(),
                       device.name().data(),
                       device.clock(),
                       device.memoryClock(),
                       device.smx(),
                       device.computeCapability(true),
                       device.computeCapability(false),
                       device.freeMemSize() / oneMiB,
                       device.globalMemSize() / oneMiB);
        }
    }


    inline void start(const Job &)
    {
        LOG_INFO("%s use profile " BLUE_BG(WHITE_BOLD_S " %s ") WHITE_BOLD_S " (" CYAN_BOLD("%zu") WHITE_BOLD(" thread%s)") " scratchpad " CYAN_BOLD("%zu KB"),
                 tag,
                 profileName.data(),
                 threads.size(),
                 threads.size() > 1 ? "s" : "",
                 algo.l3() / 1024
                 );

        Log::print(WHITE_BOLD("|  # | GPU |  BUS ID |    I |   T |   B | BF |  BS |  MEM | NAME"));

        size_t algo_l3 = algo.l3();

#       ifdef XMRIG_ALGO_ASTROBWT
        if (algo.family() == Algorithm::ASTROBWT) {
            algo_l3 = CudaAstroBWTRunner::BWT_DATA_STRIDE * 17 + 1024;
        }
#       endif

        size_t i = 0;
        for (const auto &data : threads) {
            Log::print("|" CYAN_BOLD("%3zu") " |" CYAN_BOLD("%4u") " |" YELLOW(" %7s") " |" CYAN_BOLD("%5d") " |" CYAN_BOLD("%4d") " |"
                       CYAN_BOLD("%4d") " |" CYAN_BOLD("%3d") " |" CYAN_BOLD("%4d") " |" CYAN("%5zu") " | " GREEN("%s"),
                       i,
                       data.thread.index(),
                       data.device.topology().toString().data(),
                       data.thread.threads() * data.thread.blocks(),
                       data.thread.threads(),
                       data.thread.blocks(),
                       data.thread.bfactor(),
                       data.thread.bsleep(),
                       (data.thread.threads() * data.thread.blocks()) * algo_l3 / oneMiB,
                       data.device.name().data()
                       );

                    i++;
        }

        status.start(threads.size());
        workers.start(threads);
    }


#   ifdef XMRIG_FEATURE_NVML
    void printHealth()
    {
        for (const auto &device : devices) {
            const auto health = NvmlLib::health(device.nvmlDevice());

            std::string clocks;
            if (health.clock && health.memClock) {
                clocks += " " + std::to_string(health.clock) + "/" + std::to_string(health.memClock) + " MHz";
            }

            std::string fans;
            if (!health.fanSpeed.empty()) {
                for (size_t i = 0; i < health.fanSpeed.size(); ++i) {
                    fans += " fan" + std::to_string(i) + ":" CYAN_BOLD_S + std::to_string(health.fanSpeed[i]) + "%" CLEAR;
                }
            }

            LOG_INFO("%s" CYAN_BOLD(" #%u") YELLOW(" %s") MAGENTA_BOLD("%4uW") CSI "1;%um %2uC" CLEAR WHITE_BOLD("%s") "%s",
                     tag,
                     device.index(),
                     device.topology().toString().data(),
                     health.power,
                     health.temperature < 60 ? 32 : (health.temperature > 85 ? 31 : 33),
                     health.temperature,
                     clocks.c_str(),
                     fans.c_str()
                     );
        }
    }
#   endif


    Algorithm algo;
    Controller *controller;
    CudaLaunchStatus status;
    std::vector<CudaDevice> devices;
    std::vector<CudaLaunchData> threads;
    String profileName;
    uint32_t driverVersion      = 0;
    uint32_t runtimeVersion     = 0;
    Workers<CudaLaunchData> workers;
};


} // namespace xmrig


const char *xmrig::cuda_tag()
{
    return tag;
}


xmrig::CudaBackend::CudaBackend(Controller *controller) :
    d_ptr(new CudaBackendPrivate(controller))
{
    d_ptr->workers.setBackend(this);
}


xmrig::CudaBackend::~CudaBackend()
{
    delete d_ptr;

    CudaLib::close();

#   ifdef XMRIG_FEATURE_NVML
    NvmlLib::close();
#   endif
}


bool xmrig::CudaBackend::isEnabled() const
{
    return d_ptr->controller->config()->cuda().isEnabled() && CudaLib::isInitialized() && !d_ptr->devices.empty();;
}


bool xmrig::CudaBackend::isEnabled(const Algorithm &algorithm) const
{
    return !d_ptr->controller->config()->cuda().threads().get(algorithm).isEmpty();
}


const xmrig::Hashrate *xmrig::CudaBackend::hashrate() const
{
    return d_ptr->workers.hashrate();
}


const xmrig::String &xmrig::CudaBackend::profileName() const
{
    return d_ptr->profileName;
}


const xmrig::String &xmrig::CudaBackend::type() const
{
    return kType;
}


void xmrig::CudaBackend::execCommand(char)
{
}


void xmrig::CudaBackend::prepare(const Job &)
{
}


void xmrig::CudaBackend::printHashrate(bool details)
{
    if (!details || !hashrate()) {
        return;
    }

    char num[8 * 3] = { 0 };

    Log::print(WHITE_BOLD_S "|   CUDA # | AFFINITY | 10s H/s | 60s H/s | 15m H/s |");

    size_t i = 0;
    for (const auto &data : d_ptr->threads) {
         Log::print("| %8zu | %8" PRId64 " | %7s | %7s | %7s |" CYAN_BOLD(" #%u") YELLOW(" %s") GREEN(" %s"),
                    i,
                    data.thread.affinity(),
                    Hashrate::format(hashrate()->calc(i, Hashrate::ShortInterval),  num,         sizeof num / 3),
                    Hashrate::format(hashrate()->calc(i, Hashrate::MediumInterval), num + 8,     sizeof num / 3),
                    Hashrate::format(hashrate()->calc(i, Hashrate::LargeInterval),  num + 8 * 2, sizeof num / 3),
                    data.device.index(),
                    data.device.topology().toString().data(),
                    data.device.name().data()
                    );

         i++;
    }

    Log::print(WHITE_BOLD_S "|        - |        - | %7s | %7s | %7s |",
               Hashrate::format(hashrate()->calc(Hashrate::ShortInterval),  num,         sizeof num / 3),
               Hashrate::format(hashrate()->calc(Hashrate::MediumInterval), num + 8,     sizeof num / 3),
               Hashrate::format(hashrate()->calc(Hashrate::LargeInterval),  num + 8 * 2, sizeof num / 3)
               );
}


void xmrig::CudaBackend::printHealth()
{
#   ifdef XMRIG_FEATURE_NVML
    d_ptr->printHealth();
#   endif
}


void xmrig::CudaBackend::setJob(const Job &job)
{
    const auto &cuda = d_ptr->controller->config()->cuda();
    if (cuda.isEnabled()) {
        d_ptr->init(cuda);
    }

    if (!isEnabled()) {
        return stop();
    }

    auto threads = cuda.get(d_ptr->controller->miner(), job.algorithm(), d_ptr->devices);
    if (!d_ptr->threads.empty() && d_ptr->threads.size() == threads.size() && std::equal(d_ptr->threads.begin(), d_ptr->threads.end(), threads.begin())) {
        return;
    }

    d_ptr->algo         = job.algorithm();
    d_ptr->profileName  = cuda.threads().profileName(job.algorithm());

    if (d_ptr->profileName.isNull() || threads.empty()) {
        LOG_WARN("%s " RED_BOLD("disabled") YELLOW(" (no suitable configuration found)"), tag);

        return stop();
    }

    stop();

    d_ptr->threads = std::move(threads);
    d_ptr->start(job);
}


void xmrig::CudaBackend::start(IWorker *worker, bool ready)
{
    mutex.lock();

    if (d_ptr->status.started(ready)) {
        d_ptr->status.print();

        CudaWorker::ready = true;
    }

    mutex.unlock();

    if (ready) {
        worker->start();
    }
}


void xmrig::CudaBackend::stop()
{
    if (d_ptr->threads.empty()) {
        return;
    }

    const uint64_t ts = Chrono::steadyMSecs();

    d_ptr->workers.stop();
    d_ptr->threads.clear();

    LOG_INFO("%s" YELLOW(" stopped") BLACK_BOLD(" (%" PRIu64 " ms)"), tag, Chrono::steadyMSecs() - ts);
}


void xmrig::CudaBackend::tick(uint64_t ticks)
{
    d_ptr->workers.tick(ticks);
}


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::CudaBackend::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value out(kObjectType);
    out.AddMember("type",       type().toJSON(), allocator);
    out.AddMember("enabled",    isEnabled(), allocator);
    out.AddMember("algo",       d_ptr->algo.toJSON(), allocator);
    out.AddMember("profile",    profileName().toJSON(), allocator);

    if (CudaLib::isReady()) {
        Value versions(kObjectType);
        versions.AddMember("cuda-runtime",   Value(CudaLib::version(d_ptr->runtimeVersion).c_str(), allocator), allocator);
        versions.AddMember("cuda-driver",    Value(CudaLib::version(d_ptr->driverVersion).c_str(), allocator), allocator);
        versions.AddMember("plugin",         String(CudaLib::pluginVersion()).toJSON(doc), allocator);

#       ifdef XMRIG_FEATURE_NVML
        if (NvmlLib::isReady()) {
            versions.AddMember("nvml",       StringRef(NvmlLib::version()), allocator);
            versions.AddMember("driver",     StringRef(NvmlLib::driverVersion()), allocator);
        }
#       endif

        out.AddMember("versions", versions, allocator);
    }

    if (d_ptr->threads.empty() || !hashrate()) {
        return out;
    }

    out.AddMember("hashrate", hashrate()->toJSON(doc), allocator);

    Value threads(kArrayType);

    size_t i = 0;
    for (const auto &data : d_ptr->threads) {
        Value thread = data.thread.toJSON(doc);
        thread.AddMember("hashrate", hashrate()->toJSON(i, doc), allocator);

        data.device.toJSON(thread, doc);

        i++;
        threads.PushBack(thread, allocator);
    }

    out.AddMember("threads", threads, allocator);

    return out;
}


void xmrig::CudaBackend::handleRequest(IApiRequest &)
{
}
#endif
