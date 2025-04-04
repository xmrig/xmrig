/* XMRig
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
#include "base/io/log/Tags.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Chrono.h"
#include "base/tools/String.h"
#include "core/config/Config.h"
#include "core/Controller.h"


#ifdef XMRIG_ALGO_KAWPOW
#   include "crypto/kawpow/KPCache.h"
#   include "crypto/kawpow/KPHash.h"
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
            LOG_ERR("%s " RED_BOLD("disabled") YELLOW(" (failed to start threads)"), Tags::nvidia());

            return;
        }

        LOG_INFO("%s" GREEN_BOLD(" READY") " threads " "%s%zu/%zu" BLACK_BOLD(" (%" PRIu64 " ms)"),
                 Tags::nvidia(),
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
    inline explicit CudaBackendPrivate(Controller *controller) :
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
            Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") RED_BOLD("disabled ") RED("(%s)"), kLabel, CudaLib::lastError());

            return;
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


    inline void start(const Job &job)
    {
        LOG_INFO("%s use profile " BLUE_BG(WHITE_BOLD_S " %s ") WHITE_BOLD_S " (" CYAN_BOLD("%zu") WHITE_BOLD(" thread%s)") " scratchpad " CYAN_BOLD("%zu KB"),
                 Tags::nvidia(),
                 profileName.data(),
                 threads.size(),
                 threads.size() > 1 ? "s" : "",
                 algo.l3() / 1024
                 );

        Log::print(WHITE_BOLD("|  # | GPU |  BUS ID | INTENSITY | THREADS | BLOCKS | BF |  BS | MEMORY | NAME"));

        size_t algo_l3 = algo.l3();

        size_t i = 0;
        for (const auto &data : threads) {
            size_t mem_used = (data.thread.threads() * data.thread.blocks()) * algo_l3 / oneMiB;

#           ifdef XMRIG_ALGO_KAWPOW
            if (algo.family() == Algorithm::KAWPOW) {
                const uint32_t epoch = job.height() / KPHash::EPOCH_LENGTH;
                mem_used = (KPCache::dag_size(epoch) + oneMiB - 1) / oneMiB;
            }
#           endif

            Log::print("|" CYAN_BOLD("%3zu") " |" CYAN_BOLD("%4u") " |" YELLOW(" %7s") " |" CYAN_BOLD("%10d") " |" CYAN_BOLD("%8d") " |"
                       CYAN_BOLD("%7d") " |" CYAN_BOLD("%3d") " |" CYAN_BOLD("%4d") " |" CYAN("%7zu") " | " GREEN("%s"),
                       i,
                       data.thread.index(),
                       data.device.topology().toString().data(),
                       data.thread.threads() * data.thread.blocks(),
                       data.thread.threads(),
                       data.thread.blocks(),
                       data.thread.bfactor(),
                       data.thread.bsleep(),
                       mem_used,
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
                     Tags::nvidia(),
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
    return Tags::nvidia();
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


void xmrig::CudaBackend::prepare(const Job &job)
{
    if (d_ptr) {
        d_ptr->workers.jobEarlyNotification(job);
    }
}


void xmrig::CudaBackend::printHashrate(bool details)
{
    if (!details || !hashrate()) {
        return;
    }

    char num[16 * 3] = { 0 };

    auto hashrate_short  = hashrate()->calc(Hashrate::ShortInterval);
    auto hashrate_medium = hashrate()->calc(Hashrate::MediumInterval);
    auto hashrate_large  = hashrate()->calc(Hashrate::LargeInterval);

    double scale = 1.0;
    const char* h = " H/s";

    if ((hashrate_short.second >= 1e6) || (hashrate_medium.second >= 1e6) || (hashrate_large.second >= 1e6)) {
        scale = 1e-6;

        hashrate_short.second  *= scale;
        hashrate_medium.second *= scale;
        hashrate_large.second  *= scale;

        h = "MH/s";
    }

    Log::print(WHITE_BOLD_S "|   CUDA # | AFFINITY | 10s %s | 60s %s | 15m %s |", h, h, h);

    size_t i = 0;
    for (const auto& data : d_ptr->threads) {
        auto h0 = hashrate()->calc(i, Hashrate::ShortInterval);
        auto h1 = hashrate()->calc(i, Hashrate::MediumInterval);
        auto h2 = hashrate()->calc(i, Hashrate::LargeInterval);

        h0.second *= scale;
        h1.second *= scale;
        h2.second *= scale;

        Log::print("| %8zu | %8" PRId64 " | %8s | %8s | %8s |" CYAN_BOLD(" #%u") YELLOW(" %s") GREEN(" %s"),
                    i,
                    data.thread.affinity(),
                    Hashrate::format(h0, num,          sizeof num / 3),
                    Hashrate::format(h1, num + 16,     sizeof num / 3),
                    Hashrate::format(h2, num + 16 * 2, sizeof num / 3),
                    data.device.index(),
                    data.device.topology().toString().data(),
                    data.device.name().data()
                    );

         i++;
    }

    Log::print(WHITE_BOLD_S "|        - |        - | %8s | %8s | %8s |",
               Hashrate::format(hashrate_short , num,          sizeof num / 3),
               Hashrate::format(hashrate_medium, num + 16,     sizeof num / 3),
               Hashrate::format(hashrate_large , num + 16 * 2, sizeof num / 3)
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
        LOG_WARN("%s " RED_BOLD("disabled") YELLOW(" (no suitable configuration found)"), Tags::nvidia());

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

    LOG_INFO("%s" YELLOW(" stopped") BLACK_BOLD(" (%" PRIu64 " ms)"), Tags::nvidia(), Chrono::steadyMSecs() - ts);
}


bool xmrig::CudaBackend::tick(uint64_t ticks)
{
    return d_ptr->workers.tick(ticks);
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
