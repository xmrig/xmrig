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
#include "backend/cpu/CpuBackend.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Timer.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "core/Miner.h"
#include "crypto/common/Nonce.h"


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


    inline void handleJobChange()
    {
        active = true;
        if (enabled) {
            Nonce::pause(false);;
        }

        for (IBackend *backend : backends) {
            backend->setJob(job);
        }

        if (ticks == 0) {
            ticks++;
            timer->start(500, 500);
        }
    }


    bool active         = false;
    bool enabled        = true;
    Controller *controller;
    double maxHashrate  = 0.0;
    Job job;
    std::vector<IBackend *> backends;
    Timer *timer        = nullptr;
    uint64_t ticks      = 0;
    uv_rwlock_t rwlock;
};


} // namespace xmrig



xmrig::Miner::Miner(Controller *controller)
    : d_ptr(new MinerPrivate(controller))
{
    d_ptr->timer = new Timer(this);

    d_ptr->backends.push_back(new CpuBackend(this, controller));
}


xmrig::Miner::~Miner()
{
    delete d_ptr;
}


bool xmrig::Miner::isEnabled() const
{
    return d_ptr->enabled;
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

    d_ptr->job = job;
    d_ptr->job.setIndex(index);

    Nonce::reset(index);

    uv_rwlock_wrunlock(&d_ptr->rwlock);

    d_ptr->handleJobChange();
}


void xmrig::Miner::stop()
{
//    xmrig::Handle::close(m_timer);
//    m_hashrate->stop();

    Nonce::stop();

//    for (size_t i = 0; i < m_workers.size(); ++i) {
//        m_workers[i]->join();
//    }

    for (IBackend *backend : d_ptr->backends) {
        backend->stop();
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
