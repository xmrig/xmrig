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


#include <mutex>


#include "backend/common/Hashrate.h"
#include "backend/common/interfaces/IWorker.h"
#include "backend/common/Workers.h"
#include "backend/opencl/OclBackend.h"
#include "backend/opencl/OclConfig.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Chrono.h"
#include "base/tools/String.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "rapidjson/document.h"


#ifdef XMRIG_FEATURE_API
#   include "base/api/interfaces/IApiRequest.h"
#endif


namespace xmrig {


extern template class Threads<OclThreads>;


static const char *tag          = MAGENTA_BG_BOLD(WHITE_BOLD_S " ocl ");
static const String kType       = "opencl";
constexpr const size_t oneGiB   = 1024u * 1024u * 1024u;


static void printDisabled(const char *reason)
{
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") RED_BOLD("disabled") "%s", "OPENCL", reason);
}


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
        ts        = Chrono::steadyMSecs();
    }

    size_t hugePages    = 0;
    size_t memory       = 0;
    size_t pages        = 0;
    size_t started      = 0;
    size_t threads      = 0;
    uint64_t ts         = 0;
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
            return printDisabled("");
        }

        if (!OclLib::init(cl.loader())) {
            return printDisabled(RED_S " (failed to load OpenCL runtime)");
        }

        platform = cl.platform();
        if (!platform.isValid()) {
            return printDisabled(RED_S " (selected OpenCL platform NOT found)");
        }

        devices = platform.devices();
        if (devices.empty()) {
            return printDisabled(RED_S " (no devices)");
        }

        Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("#%zu ") WHITE_BOLD("%s") "/" WHITE_BOLD("%s"), "OPENCL", platform.index(), platform.name().data(), platform.version().data());

        for (const OclDevice &device : devices) {
            const String topo = device.hasTopology() ? device.topology().toString() : "n/a";
            const size_t size = device.board().size() + device.name().size() + 64;
            char *name        = new char[size]();

            if (device.board() == device.name()) {
                snprintf(name, size, GREEN_BOLD(" %s"), device.name().data());
            }
            else {
                snprintf(name, size, GREEN_BOLD(" %s") " (" CYAN_BOLD("%s") ")", device.board().data(), device.name().data());
            }

            Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("#%zu") YELLOW(" %s") "%s " WHITE_BOLD("%uMHz") " cu:" WHITE_BOLD("%u") " mem:" CYAN("%1.2f/%1.2f") " GB", "OPENCL GPU",
                       device.index(),
                       topo.data(),
                       name,
                       device.clock(),
                       device.computeUnits(),
                       static_cast<double>(device.freeMem()) / oneGiB,
                       static_cast<double>(device.globalMem()) / oneGiB);

            delete [] name;
        }
    }


    inline void start()
    {
        LOG_INFO("%s use profile " BLUE_BG(WHITE_BOLD_S " %s ") WHITE_BOLD_S " (" CYAN_BOLD("%zu") WHITE_BOLD(" threads)") " scratchpad " CYAN_BOLD("%zu KB"),
                 tag,
                 profileName.data(),
                 threads.size(),
                 algo.l3() / 1024
                 );

        workers.stop();

        status.reset();
        status.memory   = algo.l3();
        status.threads  = threads.size();

        //workers.start(threads); // FIXME
    }


    Algorithm algo;
    Controller *controller;
    LaunchStatus status;
    OclPlatform platform;
    std::mutex mutex;
    std::vector<OclDevice> devices;
    std::vector<OclLaunchData> threads;
    String profileName;
    Workers<OclLaunchData> workers;
};


} // namespace xmrig


xmrig::OclBackend::OclBackend(Controller *controller) :
    d_ptr(new OclBackendPrivate(controller))
{
    d_ptr->workers.setBackend(this);
}


xmrig::OclBackend::~OclBackend()
{
    delete d_ptr;

    OclLib::close();
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


void xmrig::OclBackend::prepare(const Job &)
{
}


void xmrig::OclBackend::printHashrate(bool details)
{
    if (!details || !hashrate()) {
        return;
    }

    char num[8 * 3] = { 0 };

    Log::print(WHITE_BOLD_S "| OPENCL THREAD | AFFINITY | 10s H/s | 60s H/s | 15m H/s |");

    size_t i = 0;
    for (const OclLaunchData &data : d_ptr->threads) {
         Log::print("| %13zu | %8" PRId64 " | %7s | %7s | %7s |",
                    i,
                    data.thread.affinity(),
                    Hashrate::format(hashrate()->calc(i, Hashrate::ShortInterval),  num,         sizeof num / 3),
                    Hashrate::format(hashrate()->calc(i, Hashrate::MediumInterval), num + 8,     sizeof num / 3),
                    Hashrate::format(hashrate()->calc(i, Hashrate::LargeInterval),  num + 8 * 2, sizeof num / 3)
                    );

         i++;
    }
}


void xmrig::OclBackend::setJob(const Job &job)
{
    if (!isEnabled()) {
        return stop();
    }

    const OclConfig &cl = d_ptr->controller->config()->cl();

    std::vector<OclLaunchData> threads = cl.get(d_ptr->controller->miner(), job.algorithm(), d_ptr->devices, tag);
    if (!d_ptr->threads.empty() && d_ptr->threads.size() == threads.size() && std::equal(d_ptr->threads.begin(), d_ptr->threads.end(), threads.begin())) {
        return;
    }

    d_ptr->algo         = job.algorithm();
    d_ptr->profileName  = cl.threads().profileName(job.algorithm());

    if (d_ptr->profileName.isNull() || threads.empty()) {
        LOG_WARN("%s " RED_BOLD("disabled") YELLOW(" (no suitable configuration found)"), tag);

        return stop();
    }

    d_ptr->threads = std::move(threads);
    d_ptr->start();
}


void xmrig::OclBackend::start(IWorker *worker)
{
    d_ptr->mutex.lock();

    d_ptr->status.started++;

    if (d_ptr->status.started == d_ptr->status.threads) {
    }

    d_ptr->mutex.unlock();

    worker->start();
}


void xmrig::OclBackend::stop()
{
    if (d_ptr->threads.empty()) {
        return;
    }

    const uint64_t ts = Chrono::steadyMSecs();

    d_ptr->workers.stop();
    d_ptr->threads.clear();

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

    Value threads(kArrayType);
    const Hashrate *hr = hashrate();

    size_t i = 0;
    for (const OclLaunchData &data : d_ptr->threads) {
        Value thread(kObjectType);
        thread.AddMember("intensity",   data.thread.intensity(), allocator);
        thread.AddMember("affinity",    data.thread.affinity(), allocator);

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


void xmrig::OclBackend::handleRequest(IApiRequest &)
{
}
#endif
