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


#include <algorithm>
#include <uv.h>


#include "backend/common/Hashrate.h"
#include "backend/cpu/Cpu.h"
#include "backend/cpu/CpuBackend.h"
#include "base/io/log/Log.h"
#include "base/kernel/Platform.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Timer.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "core/Miner.h"
#include "crypto/common/Nonce.h"
#include "rapidjson/document.h"
#include "version.h"


#ifdef XMRIG_FEATURE_API
#   include "api/Api.h"
#   include "api/interfaces/IApiRequest.h"
#endif


namespace xmrig {


class MinerPrivate
{
public:
    inline MinerPrivate(Controller *controller) : controller(controller)
    {
        uv_rwlock_init(&rwlock);
    }


    inline ~MinerPrivate()
    {
        uv_rwlock_destroy(&rwlock);

        delete timer;

        for (IBackend *backend : backends) {
            delete backend;
        }
    }


    bool isEnabled(const Algorithm &algorithm) const
    {
        for (IBackend *backend : backends) {
            if (backend->isEnabled(algorithm)) {
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

            if (isEnabled(algo)) {
                algorithms.push_back(algo);
            }
        }
    }


    inline void handleJobChange(bool reset)
    {
        active = true;

        for (IBackend *backend : backends) {
            backend->setJob(job);
        }

        if (reset) {
            Nonce::reset(job.index());
        }
        else {
            Nonce::touch();
        }

        if (enabled) {
            Nonce::pause(false);;
        }

        if (ticks == 0) {
            ticks++;
            timer->start(500, 500);
        }
    }


#   ifdef XMRIG_FEATURE_API
    void getMiner(rapidjson::Value &reply, rapidjson::Document &doc, int version) const
    {
        using namespace rapidjson;
        auto &allocator = doc.GetAllocator();

        Value cpu(kObjectType);
        cpu.AddMember("brand",   StringRef(Cpu::info()->brand()), allocator);
        cpu.AddMember("aes",     Cpu::info()->hasAES(), allocator);
        cpu.AddMember("x64",     Cpu::info()->isX64(), allocator);
        cpu.AddMember("sockets", static_cast<uint64_t>(Cpu::info()->sockets()), allocator);

        reply.AddMember("version",      APP_VERSION, allocator);
        reply.AddMember("kind",         APP_KIND, allocator);
        reply.AddMember("ua",           StringRef(Platform::userAgent()), allocator);
        reply.AddMember("cpu",          cpu, allocator);

        if (version == 1) {
            reply.AddMember("hugepages", false, allocator);
        }

        reply.AddMember("donate_level", controller->config()->pools().donateLevel(), allocator);

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

            for (size_t i = 0; i < hr->threads(); i++) {
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
        hashrate.AddMember("highest", Hashrate::normalize(maxHashrate), allocator);

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


    Algorithms algorithms;
    bool active         = false;
    bool enabled        = true;
    Controller *controller;
    double maxHashrate  = 0.0;
    Job job;
    std::vector<IBackend *> backends;
    String userJobId;
    Timer *timer        = nullptr;
    uint64_t ticks      = 0;
    uv_rwlock_t rwlock;
};


} // namespace xmrig



xmrig::Miner::Miner(Controller *controller)
    : d_ptr(new MinerPrivate(controller))
{
    controller->addListener(this);

#   ifdef XMRIG_FEATURE_API
    controller->api()->addListener(this);
#   endif

    d_ptr->timer = new Timer(this);

    d_ptr->backends.push_back(new CpuBackend(controller));

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
    uv_rwlock_rdlock(&d_ptr->rwlock);
    Job job = d_ptr->job;
    uv_rwlock_rdunlock(&d_ptr->rwlock);

    return job;
}


void xmrig::Miner::pause()
{
    d_ptr->active = false;

    Nonce::pause(true);
    Nonce::touch();
}


void xmrig::Miner::printHashrate(bool details)
{
    char num[8 * 4] = { 0 };
    double speed[3] = { 0.0 };

    for (IBackend *backend : d_ptr->backends) {
        const Hashrate *hashrate = backend->hashrate();
        if (hashrate) {
            speed[0] += hashrate->calc(Hashrate::ShortInterval);
            speed[1] += hashrate->calc(Hashrate::MediumInterval);
            speed[2] += hashrate->calc(Hashrate::LargeInterval);
        }

        backend->printHashrate(details);
    }

    LOG_INFO(WHITE_BOLD("speed") " 10s/60s/15m " CYAN_BOLD("%s") CYAN(" %s %s ") CYAN_BOLD("H/s") " max " CYAN_BOLD("%s H/s"),
             Hashrate::format(speed[0],             num,         sizeof(num) / 4),
             Hashrate::format(speed[1],             num + 8,     sizeof(num) / 4),
             Hashrate::format(speed[2],             num + 8 * 2, sizeof(num) / 4 ),
             Hashrate::format(d_ptr->maxHashrate,   num + 8 * 3, sizeof(num) / 4)
             );
}


void xmrig::Miner::setEnabled(bool enabled)
{
    if (d_ptr->enabled == enabled) {
        return;
    }

    d_ptr->enabled = enabled;

    if (enabled) {
        LOG_INFO(GREEN_BOLD("resumed"));
    }
    else {
        LOG_INFO(YELLOW_BOLD("paused") ", press " MAGENTA_BG_BOLD(" r ") " to resume");
    }

    if (!d_ptr->active) {
        return;
    }

    Nonce::pause(!enabled);
    Nonce::touch();
}


void xmrig::Miner::setJob(const Job &job, bool donate)
{
    uv_rwlock_wrlock(&d_ptr->rwlock);

    const uint8_t index = donate ? 1 : 0;
    const bool reset    = !(d_ptr->job.index() == 1 && index == 0 && d_ptr->userJobId == job.id());

    d_ptr->job = job;
    d_ptr->job.setIndex(index);

    if (index == 0) {
        d_ptr->userJobId = job.id();
    }

    uv_rwlock_wrunlock(&d_ptr->rwlock);

    d_ptr->handleJobChange(reset);
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
    double maxHashrate = 0.0;

    for (IBackend *backend : d_ptr->backends) {
        backend->tick(d_ptr->ticks);

        if (backend->hashrate()) {
            maxHashrate += backend->hashrate()->calc(Hashrate::ShortInterval);
        }
    }

    d_ptr->maxHashrate = std::max(d_ptr->maxHashrate, maxHashrate);

    if ((d_ptr->ticks % (d_ptr->controller->config()->printTime() * 2)) == 0) {
        printHashrate(false);
    }

    d_ptr->ticks++;
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
}
#endif
