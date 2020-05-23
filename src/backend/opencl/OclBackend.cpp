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


#include "backend/opencl/OclBackend.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/common/Hashrate.h"
#include "backend/common/interfaces/IWorker.h"
#include "backend/common/Tags.h"
#include "backend/common/Workers.h"
#include "backend/opencl/OclConfig.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/OclWorker.h"
#include "backend/opencl/runners/tools/OclSharedState.h"
#include "backend/opencl/runners/OclAstroBWTRunner.h"
#include "backend/opencl/wrappers/OclContext.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Chrono.h"
#include "base/tools/String.h"
#include "core/config/Config.h"
#include "core/Controller.h"


#ifdef XMRIG_FEATURE_API
#   include "base/api/interfaces/IApiRequest.h"
#endif


#ifdef XMRIG_FEATURE_ADL
#include "backend/opencl/wrappers/AdlLib.h"

namespace xmrig { static const char *kAdlLabel = "ADL"; }
#endif


namespace xmrig {


extern template class Threads<OclThreads>;


constexpr const size_t oneMiB   = 1024U * 1024U;
static const char *kLabel       = "OPENCL";
static const char *tag          = MAGENTA_BG_BOLD(WHITE_BOLD_S " ocl ");
static const String kType       = "opencl";
static std::mutex mutex;


static void printDisabled(const char *label, const char *reason)
{
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") RED_BOLD("disabled") "%s", label, reason);
}


struct OclLaunchStatus
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
        m_started        = 0;
        m_errors         = 0;
        m_threads        = threads;
        m_ts             = Chrono::steadyMSecs();
        OclWorker::ready = false;
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


class OclBackendPrivate
{
public:
    inline OclBackendPrivate(Controller *controller) :
        controller(controller)
    {
        init(controller->config()->cl());
    }


    void init(const OclConfig &cl)
    {
        if (!cl.isEnabled()) {
            return printDisabled(kLabel, "");
        }

        if (!OclLib::init(cl.loader())) {
            return printDisabled(kLabel, RED_S " (failed to load OpenCL runtime)");
        }

        if (platform.isValid()) {
            return;
        }

        platform = cl.platform();
        if (!platform.isValid()) {
            return printDisabled(kLabel, RED_S " (selected OpenCL platform NOT found)");
        }

        devices = platform.devices();
        if (devices.empty()) {
            return printDisabled(kLabel, RED_S " (no devices)");
        }

#       ifdef XMRIG_FEATURE_ADL
        if (cl.isAdlEnabled()) {
            if (AdlLib::init()) {
                Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") "press " MAGENTA_BG(WHITE_BOLD_S "e") " for health report",
                           kAdlLabel
                           );
            }
            else {
                printDisabled(kAdlLabel, RED_S " (failed to load ADL)");
            }
        }
        else {
            printDisabled(kAdlLabel, "");
        }
#       endif

        Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("#%zu ") WHITE_BOLD("%s") "/" WHITE_BOLD("%s"), "OPENCL", platform.index(), platform.name().data(), platform.version().data());

        for (const OclDevice &device : devices) {
            Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("#%zu") YELLOW(" %s") " %s " WHITE_BOLD("%u MHz") " cu:" WHITE_BOLD("%u") " mem:" CYAN("%zu/%zu") " MB",
                       "OPENCL GPU",
                       device.index(),
                       device.topology().toString().data(),
                       device.printableName().data(),
                       device.clock(),
                       device.computeUnits(),
                       device.freeMemSize() / oneMiB,
                       device.globalMemSize() / oneMiB);
        }
    }


    inline void start(const Job &job)
    {
        LOG_INFO("%s use profile " BLUE_BG(WHITE_BOLD_S " %s ") WHITE_BOLD_S " (" CYAN_BOLD("%zu") WHITE_BOLD(" thread%s)") " scratchpad " CYAN_BOLD("%zu KB"),
                 tag,
                 profileName.data(),
                 threads.size(),
                 threads.size() > 1 ? "s" : "",
                 algo.l3() / 1024
                 );

        Log::print(WHITE_BOLD("|  # | GPU |  BUS ID |    I |  W | SI | MC |  U |  MEM | NAME"));

        size_t algo_l3 = algo.l3();

#       ifdef XMRIG_ALGO_ASTROBWT
        if (algo.family() == Algorithm::ASTROBWT) {
            algo_l3 = OclAstroBWTRunner::BWT_DATA_STRIDE * 17 + 324;
        }
#       endif

        size_t i = 0;
        for (const auto &data : threads) {
            Log::print("|" CYAN_BOLD("%3zu") " |" CYAN_BOLD("%4u") " |" YELLOW(" %7s") " |" CYAN_BOLD("%5u") " |" CYAN_BOLD("%3u") " |"
                       CYAN_BOLD("%3u") " |" CYAN_BOLD("%3s") " |" CYAN_BOLD("%3u") " |" CYAN("%5zu") " | %s",
                       i,
                       data.thread.index(),
                       data.device.topology().toString().data(),
                       data.thread.intensity(),
                       data.thread.worksize(),
                       data.thread.stridedIndex(),
                       data.thread.stridedIndex() == 2 ? std::to_string(data.thread.memChunk()).c_str() : "-",
                       data.thread.unrollFactor(),
                       data.thread.intensity() * algo_l3 / oneMiB,
                       data.device.printableName().data()
                       );

                    i++;
        }

        OclSharedState::start(threads, job);

        status.start(threads.size());
        workers.start(threads);
    }


#   ifdef XMRIG_FEATURE_ADL
    void printHealth()
    {
        if (!AdlLib::isReady()) {
            return;
        }

        for (const auto &device : devices) {
            const auto health = AdlLib::health(device);

            LOG_INFO("%s" CYAN_BOLD(" #%u") YELLOW(" %s") MAGENTA_BOLD("%4uW") CSI "1;%um %2uC" CYAN_BOLD(" %4u") CYAN("RPM") WHITE_BOLD(" %u/%u") "MHz",
                     tag,
                     device.index(),
                     device.topology().toString().data(),
                     health.power,
                     health.temperature < 60 ? 32 : (health.temperature > 85 ? 31 : 33),
                     health.temperature,
                     health.rpm,
                     health.clock,
                     health.memClock
                     );
        }
    }
#   endif


    Algorithm algo;
    Controller *controller;
    OclContext context;
    OclLaunchStatus status;
    OclPlatform platform;
    std::vector<OclDevice> devices;
    std::vector<OclLaunchData> threads;
    String profileName;
    Workers<OclLaunchData> workers;
};


} // namespace xmrig


const char *xmrig::ocl_tag()
{
    return tag;
}


xmrig::OclBackend::OclBackend(Controller *controller) :
    d_ptr(new OclBackendPrivate(controller))
{
    d_ptr->workers.setBackend(this);
}


xmrig::OclBackend::~OclBackend()
{
    delete d_ptr;

    OclLib::close();

#   ifdef XMRIG_FEATURE_ADL
    AdlLib::close();
#   endif
}


bool xmrig::OclBackend::isEnabled() const
{
    return d_ptr->controller->config()->cl().isEnabled() && OclLib::isInitialized() && d_ptr->platform.isValid() && !d_ptr->devices.empty();
}


bool xmrig::OclBackend::isEnabled(const Algorithm &algorithm) const
{
    return !d_ptr->controller->config()->cl().threads().get(algorithm).isEmpty();
}


const xmrig::Hashrate *xmrig::OclBackend::hashrate() const
{
    return d_ptr->workers.hashrate();
}


const xmrig::String &xmrig::OclBackend::profileName() const
{
    return d_ptr->profileName;
}


const xmrig::String &xmrig::OclBackend::type() const
{
    return kType;
}


void xmrig::OclBackend::execCommand(char)
{
}


void xmrig::OclBackend::prepare(const Job &)
{
}


void xmrig::OclBackend::printHashrate(bool details)
{
    if (!details || !hashrate()) {
        return;
    }

    char num[8 * 3] = { 0 };

    Log::print(WHITE_BOLD_S "| OPENCL # | AFFINITY | 10s H/s | 60s H/s | 15m H/s |");

    size_t i = 0;
    for (const auto &data : d_ptr->threads) {
         Log::print("| %8zu | %8" PRId64 " | %7s | %7s | %7s |" CYAN_BOLD(" #%u") YELLOW(" %s") " %s",
                    i,
                    data.affinity,
                    Hashrate::format(hashrate()->calc(i, Hashrate::ShortInterval),  num,         sizeof num / 3),
                    Hashrate::format(hashrate()->calc(i, Hashrate::MediumInterval), num + 8,     sizeof num / 3),
                    Hashrate::format(hashrate()->calc(i, Hashrate::LargeInterval),  num + 8 * 2, sizeof num / 3),
                    data.device.index(),
                    data.device.topology().toString().data(),
                    data.device.printableName().data()
                    );

         i++;
    }

    Log::print(WHITE_BOLD_S "|        - |        - | %7s | %7s | %7s |",
               Hashrate::format(hashrate()->calc(Hashrate::ShortInterval),  num,         sizeof num / 3),
               Hashrate::format(hashrate()->calc(Hashrate::MediumInterval), num + 8,     sizeof num / 3),
               Hashrate::format(hashrate()->calc(Hashrate::LargeInterval),  num + 8 * 2, sizeof num / 3)
               );
}


void xmrig::OclBackend::printHealth()
{
#   ifdef XMRIG_FEATURE_ADL
    d_ptr->printHealth();
#   endif
}


void xmrig::OclBackend::setJob(const Job &job)
{
    const auto &cl = d_ptr->controller->config()->cl();
    if (cl.isEnabled()) {
        d_ptr->init(cl);
    }

    if (!isEnabled()) {
        return stop();
    }

    auto threads = cl.get(d_ptr->controller->miner(), job.algorithm(), d_ptr->platform, d_ptr->devices);
    if (!d_ptr->threads.empty() && d_ptr->threads.size() == threads.size() && std::equal(d_ptr->threads.begin(), d_ptr->threads.end(), threads.begin())) {
        return;
    }

    d_ptr->algo         = job.algorithm();
    d_ptr->profileName  = cl.threads().profileName(job.algorithm());

    if (d_ptr->profileName.isNull() || threads.empty()) {
        LOG_WARN("%s " RED_BOLD("disabled") YELLOW(" (no suitable configuration found)"), tag);

        return stop();
    }

    if (!d_ptr->context.init(d_ptr->devices, threads)) {
        LOG_WARN("%s " RED_BOLD("disabled") YELLOW(" (OpenCL context unavailable)"), tag);

        return stop();
    }

    stop();

    d_ptr->threads = std::move(threads);
    d_ptr->start(job);
}


void xmrig::OclBackend::start(IWorker *worker, bool ready)
{
    mutex.lock();

    if (d_ptr->status.started(ready)) {
        d_ptr->status.print();

        OclWorker::ready = true;
    }

    mutex.unlock();

    if (ready) {
        worker->start();
    }
}


void xmrig::OclBackend::stop()
{
    if (d_ptr->threads.empty()) {
        return;
    }

    const uint64_t ts = Chrono::steadyMSecs();

    d_ptr->workers.stop();
    d_ptr->threads.clear();

    OclSharedState::release();

    LOG_INFO("%s" YELLOW(" stopped") BLACK_BOLD(" (%" PRIu64 " ms)"), tag, Chrono::steadyMSecs() - ts);
}


void xmrig::OclBackend::tick(uint64_t ticks)
{
    d_ptr->workers.tick(ticks);
}


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::OclBackend::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value out(kObjectType);
    out.AddMember("type",       type().toJSON(), allocator);
    out.AddMember("enabled",    isEnabled(), allocator);
    out.AddMember("algo",       d_ptr->algo.toJSON(), allocator);
    out.AddMember("profile",    profileName().toJSON(), allocator);
    out.AddMember("platform",   d_ptr->platform.toJSON(doc), allocator);

    if (d_ptr->threads.empty() || !hashrate()) {
        return out;
    }

    out.AddMember("hashrate", hashrate()->toJSON(doc), allocator);

    Value threads(kArrayType);

    size_t i = 0;
    for (const auto &data : d_ptr->threads) {
        Value thread = data.thread.toJSON(doc);
        thread.AddMember("affinity", data.affinity, allocator);
        thread.AddMember("hashrate", hashrate()->toJSON(i, doc), allocator);

        data.device.toJSON(thread, doc);

        i++;
        threads.PushBack(thread, allocator);
    }

    out.AddMember("threads", threads, allocator);

    return out;
}


void xmrig::OclBackend::handleRequest(IApiRequest &)
{
}
#endif
