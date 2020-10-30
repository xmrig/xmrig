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


#include <algorithm>
#include <mutex>
#include <thread>


#include "core/Miner.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/common/Hashrate.h"
#include "backend/cpu/Cpu.h"
#include "backend/cpu/CpuBackend.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/kernel/Platform.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Object.h"
#include "base/tools/Profiler.h"
#include "base/tools/Timer.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "crypto/common/Nonce.h"
#include "crypto/rx/Rx.h"
#include "version.h"


#ifdef XMRIG_FEATURE_API
#   include "base/api/Api.h"
#   include "base/api/interfaces/IApiRequest.h"
#endif


#ifdef XMRIG_FEATURE_OPENCL
#   include "backend/opencl/OclBackend.h"
#endif


#ifdef XMRIG_FEATURE_CUDA
#   include "backend/cuda/CudaBackend.h"
#endif


#ifdef XMRIG_ALGO_RANDOMX
#   include "crypto/rx/RxConfig.h"
#endif


#ifdef XMRIG_ALGO_ASTROBWT
#   include "crypto/astrobwt/AstroBWT.h"
#endif


namespace xmrig {


static std::mutex mutex;


class MinerPrivate
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(MinerPrivate)


    inline MinerPrivate(Controller *controller) : controller(controller) {}


    inline ~MinerPrivate()
    {
        delete timer;

        for (IBackend *backend : backends) {
            delete backend;
        }

#       ifdef XMRIG_ALGO_RANDOMX
        Rx::destroy();
#       endif
    }


    bool isEnabled(const Algorithm &algorithm) const
    {
        for (IBackend *backend : backends) {
            if (backend->isEnabled() && backend->isEnabled(algorithm)) {
                return true;
            }
        }

        return false;
    }


    inline void rebuild()
    {
        algorithms.clear();

        for (int i = 0; i < Algorithm::MAX; ++i) {
            const Algorithm algo(static_cast<Algorithm::Id>(i));

            if (algo.isValid() && isEnabled(algo)) {
                algorithms.push_back(algo);
            }
        }
    }


    inline void handleJobChange()
    {
        if (!enabled) {
            Nonce::pause(true);
        }

        if (reset) {
            Nonce::reset(job.index());
        }

        for (IBackend *backend : backends) {
            backend->setJob(job);
        }

        Nonce::touch();

        if (active && enabled) {
            Nonce::pause(false);
        }

        if (ticks == 0) {
            ticks++;
            timer->start(500, 500);
        }
    }


#   ifdef XMRIG_FEATURE_API
    void getMiner(rapidjson::Value &reply, rapidjson::Document &doc, int) const
    {
        using namespace rapidjson;
        auto &allocator = doc.GetAllocator();

        reply.AddMember("version",      APP_VERSION, allocator);
        reply.AddMember("kind",         APP_KIND, allocator);
        reply.AddMember("ua",           StringRef(Platform::userAgent()), allocator);
        reply.AddMember("cpu",          Cpu::toJSON(doc), allocator);
        reply.AddMember("donate_level", controller->config()->pools().donateLevel(), allocator);
        reply.AddMember("paused",       !enabled, allocator);

        Value algo(kArrayType);

        for (const Algorithm &a : algorithms) {
            algo.PushBack(StringRef(a.shortName()), allocator);
        }

        reply.AddMember("algorithms", algo, allocator);
    }


    void getHashrate(rapidjson::Value &reply, rapidjson::Document &doc, int version) const
    {
        using namespace rapidjson;
        auto &allocator = doc.GetAllocator();

        Value hashrate(kObjectType);
        Value total(kArrayType);
        Value threads(kArrayType);

        double t[3] = { 0.0 };

        for (IBackend *backend : backends) {
            const Hashrate *hr = backend->hashrate();
            if (!hr) {
                continue;
            }

            t[0] += hr->calc(Hashrate::ShortInterval);
            t[1] += hr->calc(Hashrate::MediumInterval);
            t[2] += hr->calc(Hashrate::LargeInterval);

            if (version > 1) {
                continue;
            }

            for (size_t i = 1; i < hr->threads(); i++) {
                Value thread(kArrayType);
                thread.PushBack(Hashrate::normalize(hr->calc(i, Hashrate::ShortInterval)),  allocator);
                thread.PushBack(Hashrate::normalize(hr->calc(i, Hashrate::MediumInterval)), allocator);
                thread.PushBack(Hashrate::normalize(hr->calc(i, Hashrate::LargeInterval)),  allocator);

                threads.PushBack(thread, allocator);
            }
        }

        total.PushBack(Hashrate::normalize(t[0]),  allocator);
        total.PushBack(Hashrate::normalize(t[1]), allocator);
        total.PushBack(Hashrate::normalize(t[2]),  allocator);

        hashrate.AddMember("total",   total, allocator);
        hashrate.AddMember("highest", Hashrate::normalize(maxHashrate[algorithm]), allocator);

        if (version == 1) {
            hashrate.AddMember("threads", threads, allocator);
        }

        reply.AddMember("hashrate", hashrate, allocator);
    }


    void getBackends(rapidjson::Value &reply, rapidjson::Document &doc) const
    {
        using namespace rapidjson;
        auto &allocator = doc.GetAllocator();

        reply.SetArray();

        for (IBackend *backend : backends) {
            reply.PushBack(backend->toJSON(doc), allocator);
        }
    }
#   endif


    static inline void printProfile()
    {
#       ifdef XMRIG_FEATURE_PROFILING
        ProfileScopeData* data[ProfileScopeData::MAX_DATA_COUNT];

        const uint32_t n = std::min<uint32_t>(ProfileScopeData::s_dataCount, ProfileScopeData::MAX_DATA_COUNT);
        memcpy(data, ProfileScopeData::s_data, n * sizeof(ProfileScopeData*));

        std::sort(data, data + n, [](ProfileScopeData* a, ProfileScopeData* b) {
            return strcmp(a->m_threadId, b->m_threadId) < 0;
        });

        for (uint32_t i = 0; i < n;)
        {
            uint32_t n1 = i;
            while ((n1 < n) && (strcmp(data[i]->m_threadId, data[n1]->m_threadId) == 0)) {
                ++n1;
            }

            std::sort(data + i, data + n1, [](ProfileScopeData* a, ProfileScopeData* b) {
                return a->m_totalCycles > b->m_totalCycles;
            });

            for (uint32_t j = i; j < n1; ++j) {
                ProfileScopeData* p = data[j];
                LOG_INFO("%s Thread %6s | %-30s | %7.3f%% | %9.0f ns",
                    Tags::profiler(),
                    p->m_threadId,
                    p->m_name,
                    p->m_totalCycles * 100.0 / data[i]->m_totalCycles,
                    p->m_totalCycles / p->m_totalSamples * 1e9 / ProfileScopeData::s_tscSpeed
                );
            }

            LOG_INFO("%s --------------|--------------------------------|----------|-------------", Tags::profiler());

            i = n1;
        }
#       endif
    }


    void printHashrate(bool details)
    {
        char num[16 * 4] = { 0 };
        double speed[3]  = { 0.0 };
        uint32_t count   = 0;

        for (auto backend : backends) {
            const auto hashrate = backend->hashrate();
            if (hashrate) {
                ++count;

                speed[0] += hashrate->calc(Hashrate::ShortInterval);
                speed[1] += hashrate->calc(Hashrate::MediumInterval);
                speed[2] += hashrate->calc(Hashrate::LargeInterval);
            }

            backend->printHashrate(details);
        }

        if (!count) {
            return;
        }

        printProfile();

        double scale  = 1.0;
        const char* h = "H/s";

        if ((speed[0] >= 1e6) || (speed[1] >= 1e6) || (speed[2] >= 1e6) || (maxHashrate[algorithm] >= 1e6)) {
            scale = 1e-6;
            h = "MH/s";
        }

        LOG_INFO("%s " WHITE_BOLD("speed") " 10s/60s/15m " CYAN_BOLD("%s") CYAN(" %s %s ") CYAN_BOLD("%s") " max " CYAN_BOLD("%s %s"),
                 Tags::miner(),
                 Hashrate::format(speed[0] * scale,                 num,          sizeof(num) / 4),
                 Hashrate::format(speed[1] * scale,                 num + 16,     sizeof(num) / 4),
                 Hashrate::format(speed[2] * scale,                 num + 16 * 2, sizeof(num) / 4), h,
                 Hashrate::format(maxHashrate[algorithm] * scale,   num + 16 * 3, sizeof(num) / 4), h
                 );

#       ifdef XMRIG_FEATURE_BENCHMARK
        for (auto backend : backends) {
            backend->printBenchProgress();
        }
#       endif
    }


#   ifdef XMRIG_ALGO_RANDOMX
    inline bool initRX() { return Rx::init(job, controller->config()->rx(), controller->config()->cpu()); }
#   endif


    Algorithm algorithm;
    Algorithms algorithms;
    bool active         = false;
    bool battery_power  = false;
    bool enabled        = true;
    bool reset          = true;
    Controller *controller;
    Job job;
    mutable std::map<Algorithm::Id, double> maxHashrate;
    std::vector<IBackend *> backends;
    String userJobId;
    Timer *timer        = nullptr;
    uint64_t ticks      = 0;
};


} // namespace xmrig



xmrig::Miner::Miner(Controller *controller)
    : d_ptr(new MinerPrivate(controller))
{
    const int priority = controller->config()->cpu().priority();
    if (priority >= 0) {
        Platform::setProcessPriority(priority);
        Platform::setThreadPriority(std::min(priority + 1, 5));
    }

#   ifdef XMRIG_FEATURE_PROFILING
    ProfileScopeData::Init();
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    Rx::init(this);
#   endif

#   ifdef XMRIG_ALGO_ASTROBWT
    astrobwt::init();
#   endif

    controller->addListener(this);

#   ifdef XMRIG_FEATURE_API
    controller->api()->addListener(this);
#   endif

    d_ptr->timer = new Timer(this);

    d_ptr->backends.reserve(3);
    d_ptr->backends.push_back(new CpuBackend(controller));

#   ifdef XMRIG_FEATURE_OPENCL
    d_ptr->backends.push_back(new OclBackend(controller));
#   endif

#   ifdef XMRIG_FEATURE_CUDA
    d_ptr->backends.push_back(new CudaBackend(controller));
#   endif

    d_ptr->rebuild();
}


xmrig::Miner::~Miner()
{
    delete d_ptr;
}


bool xmrig::Miner::isEnabled() const
{
    return d_ptr->enabled;
}


bool xmrig::Miner::isEnabled(const Algorithm &algorithm) const
{
    return std::find(d_ptr->algorithms.begin(), d_ptr->algorithms.end(), algorithm) != d_ptr->algorithms.end();
}


const xmrig::Algorithms &xmrig::Miner::algorithms() const
{
    return d_ptr->algorithms;
}


const std::vector<xmrig::IBackend *> &xmrig::Miner::backends() const
{
    return d_ptr->backends;
}


xmrig::Job xmrig::Miner::job() const
{
    std::lock_guard<std::mutex> lock(mutex);

    return d_ptr->job;
}


void xmrig::Miner::execCommand(char command)
{
    switch (command) {
    case 'h':
    case 'H':
        d_ptr->printHashrate(true);
        break;

    case 'p':
    case 'P':
        setEnabled(false);
        break;

    case 'r':
    case 'R':
        setEnabled(true);
        break;

    case 'e':
    case 'E':
        for (auto backend : d_ptr->backends) {
            backend->printHealth();
        }
        break;

    default:
        break;
    }

    for (auto backend : d_ptr->backends) {
        backend->execCommand(command);
    }
}


void xmrig::Miner::pause()
{
    d_ptr->active = false;

    Nonce::pause(true);
    Nonce::touch();
}


void xmrig::Miner::setEnabled(bool enabled)
{
    if (d_ptr->enabled == enabled) {
        return;
    }

    if (d_ptr->battery_power && enabled) {
        LOG_INFO("%s " YELLOW_BOLD("can't resume while on battery power"), Tags::miner());

        return;
    }

    d_ptr->enabled = enabled;

    if (enabled) {
        LOG_INFO("%s " GREEN_BOLD("resumed"), Tags::miner());
    }
    else {
        if (d_ptr->battery_power) {
            LOG_INFO("%s " YELLOW_BOLD("paused"), Tags::miner());
        }
        else {
            LOG_INFO("%s " YELLOW_BOLD("paused") ", press " MAGENTA_BG_BOLD(" r ") " to resume", Tags::miner());
        }
    }

    if (!d_ptr->active) {
        return;
    }

    Nonce::pause(!enabled);
    Nonce::touch();
}


void xmrig::Miner::setJob(const Job &job, bool donate)
{
    for (IBackend *backend : d_ptr->backends) {
        backend->prepare(job);
    }

#   ifdef XMRIG_ALGO_RANDOMX
    if (job.algorithm().family() == Algorithm::RANDOM_X && !Rx::isReady(job)) {
        stop();
    }
#   endif

    d_ptr->algorithm = job.algorithm();

    mutex.lock();

    const uint8_t index = donate ? 1 : 0;

    d_ptr->reset = !(d_ptr->job.index() == 1 && index == 0 && d_ptr->userJobId == job.id());
    d_ptr->job   = job;
    d_ptr->job.setIndex(index);

    if (index == 0) {
        d_ptr->userJobId = job.id();
    }

#   ifdef XMRIG_ALGO_RANDOMX
    const bool ready = d_ptr->initRX();
#   else
    constexpr const bool ready = true;
#   endif

    mutex.unlock();

    d_ptr->active = true;

    if (ready) {
        d_ptr->handleJobChange();
    }
}


void xmrig::Miner::stop()
{
    Nonce::stop();

    for (IBackend *backend : d_ptr->backends) {
        backend->stop();
    }
}


void xmrig::Miner::onConfigChanged(Config *config, Config *previousConfig)
{
    d_ptr->rebuild();

    if (config->pools() != previousConfig->pools() && config->pools().active() > 0) {
        return;
    }

    const Job job = this->job();

    for (IBackend *backend : d_ptr->backends) {
        backend->setJob(job);
    }
}


void xmrig::Miner::onTimer(const Timer *)
{
    double maxHashrate          = 0.0;
    const auto healthPrintTime  = d_ptr->controller->config()->healthPrintTime();

    bool stopMiner = false;

    for (IBackend *backend : d_ptr->backends) {
        if (!backend->tick(d_ptr->ticks)) {
            stopMiner = true;
        }

        if (healthPrintTime && d_ptr->ticks && (d_ptr->ticks % (healthPrintTime * 2)) == 0 && backend->isEnabled()) {
            backend->printHealth();
        }

        if (backend->hashrate()) {
            maxHashrate += backend->hashrate()->calc(Hashrate::ShortInterval);
        }
    }

    d_ptr->maxHashrate[d_ptr->algorithm] = std::max(d_ptr->maxHashrate[d_ptr->algorithm], maxHashrate);

    const auto printTime = d_ptr->controller->config()->printTime();
    if (printTime && d_ptr->ticks && (d_ptr->ticks % (printTime * 2)) == 0) {
        d_ptr->printHashrate(false);
    }

    d_ptr->ticks++;

    if (d_ptr->controller->config()->isPauseOnBattery()) {
        const bool battery_power = Platform::isOnBatteryPower();
        if (battery_power && d_ptr->enabled) {
            LOG_INFO("%s " YELLOW_BOLD("on battery power"), Tags::miner());
            d_ptr->battery_power = true;
            setEnabled(false);
        }
        else if (!battery_power && !d_ptr->enabled && d_ptr->battery_power) {
            LOG_INFO("%s " GREEN_BOLD("on AC power"), Tags::miner());
            d_ptr->battery_power = false;
            setEnabled(true);
        }
    }

    if (stopMiner) {
        stop();
    }
}


#ifdef XMRIG_FEATURE_API
void xmrig::Miner::onRequest(IApiRequest &request)
{
    if (request.method() == IApiRequest::METHOD_GET) {
        if (request.type() == IApiRequest::REQ_SUMMARY) {
            request.accept();

            d_ptr->getMiner(request.reply(), request.doc(), request.version());
            d_ptr->getHashrate(request.reply(), request.doc(), request.version());
        }
        else if (request.url() == "/2/backends") {
            request.accept();

            d_ptr->getBackends(request.reply(), request.doc());
        }
    }
    else if (request.type() == IApiRequest::REQ_JSON_RPC) {
        if (request.rpcMethod() == "pause") {
            request.accept();

            setEnabled(false);
        }
        else if (request.rpcMethod() == "resume") {
            request.accept();

            setEnabled(true);
        }
        else if (request.rpcMethod() == "stop") {
            request.accept();

            stop();
        }
    }

    for (IBackend *backend : d_ptr->backends) {
        backend->handleRequest(request);
    }
}
#endif


#ifdef XMRIG_ALGO_RANDOMX
void xmrig::Miner::onDatasetReady()
{
    if (!Rx::isReady(job())) {
        return;
    }

    d_ptr->handleJobChange();
}
#endif
