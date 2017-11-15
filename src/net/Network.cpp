/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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

#ifdef _MSC_VER
#pragma warning(disable:4244)
#endif

#include <inttypes.h>
#include <memory>
#include <time.h>


#include "api/Api.h"
#include "log/Log.h"
#include "net/Client.h"
#include "net/Network.h"
#include "net/strategies/DonateStrategy.h"
#include "net/strategies/FailoverStrategy.h"
#include "net/strategies/SinglePoolStrategy.h"
#include "net/SubmitResult.h"
#include "net/Url.h"
#include "Options.h"
#include "Platform.h"
#include "workers/Workers.h"


Network::Network(const Options *options) :
    m_options(options),
    m_donate(nullptr)
{
    srand(time(0) ^ (uintptr_t) this);

    Workers::setListener(this);

    const std::vector<Url*> &pools = options->pools();

    if (pools.size() > 1) {
        m_strategy = new FailoverStrategy(pools, Platform::userAgent(), this);
    }
    else {
        m_strategy = new SinglePoolStrategy(pools.front(), Platform::userAgent(), this);
    }

    if (m_options->donateLevel() > 0) {
        m_donate = new DonateStrategy(Platform::userAgent(), this);
    }

    m_timer.data = this;
    uv_timer_init(uv_default_loop(), &m_timer);

    uv_timer_start(&m_timer, Network::onTick, kTickInterval, kTickInterval);
}


Network::~Network()
{
}


void Network::connect()
{
    m_strategy->connect();
}


void Network::stop()
{
    if (m_donate) {
        m_donate->stop();
    }

    m_strategy->stop();
}


void Network::onActive(Client *client)
{
    if (client->id() == -1) {
        LOG_NOTICE("dev donate started");
        return;
    }

    m_state.setPool(client->host(), client->port(), client->ip());

    LOG_INFO(m_options->colors() ? "\x1B[01;37muse pool \x1B[01;36m%s:%d \x1B[01;30m%s" : "use pool %s:%d %s", client->host(), client->port(), client->ip());
}


void Network::onJob(Client *client, const Job &job)
{
    if (m_donate && m_donate->isActive() && client->id() != -1) {
        return;
    }

    setJob(client, job);
}


void Network::onJobResult(const JobResult &result)
{
    if (result.poolId == -1 && m_donate) {
        m_donate->submit(result);
        return;
    }

    m_strategy->submit(result);
}


void Network::onPause(IStrategy *strategy)
{
    if (m_donate && m_donate == strategy) {
        LOG_NOTICE("dev donate finished");
        m_strategy->resume();
    }

    if (!m_strategy->isActive()) {
        LOG_ERR("no active pools, stop mining");
        m_state.stop();
        return Workers::pause();
    }
}


void Network::onResultAccepted(Client *client, const SubmitResult &result, const char *error)
{
    m_state.add(result, error);

    if (error) {
        LOG_INFO(m_options->colors() ? "\x1B[01;31mrejected\x1B[0m (%" PRId64 "/%" PRId64 ") diff \x1B[01;37m%u\x1B[0m \x1B[31m\"%s\"\x1B[0m \x1B[01;30m(%" PRIu64 " ms)"
                                     : "rejected (%" PRId64 "/%" PRId64 ") diff %u \"%s\" (%" PRIu64 " ms)",
                 m_state.accepted, m_state.rejected, result.diff, error, result.elapsed);
    }
    else {
        LOG_INFO(m_options->colors() ? "\x1B[01;32maccepted\x1B[0m (%" PRId64 "/%" PRId64 ") diff \x1B[01;37m%u\x1B[0m \x1B[01;30m(%" PRIu64 " ms)"
                                     : "accepted (%" PRId64 "/%" PRId64 ") diff %u (%" PRIu64 " ms)",
                 m_state.accepted, m_state.rejected, result.diff, result.elapsed);
    }
}


void Network::setJob(Client *client, const Job &job)
{
    if (m_options->colors()) {
        LOG_INFO("\x1B[01;35mnew job\x1B[0m from \x1B[01;37m%s:%d\x1B[0m diff \x1B[01;37m%d", client->host(), client->port(), job.diff());
    }
    else {
        LOG_INFO("new job from %s:%d diff %d", client->host(), client->port(), job.diff());
    }

    m_state.diff = job.diff();
    Workers::setJob(job);
}


void Network::tick()
{
    const uint64_t now = uv_now(uv_default_loop());

    m_strategy->tick(now);

    if (m_donate) {
        m_donate->tick(now);
    }

#   ifndef XMRIG_NO_API
    Api::tick(m_state);
#   endif
}


void Network::onTick(uv_timer_t *handle)
{
    static_cast<Network*>(handle->data)->tick();
}
