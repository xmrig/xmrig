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


#include <memory>


#include "log/Log.h"
#include "net/Client.h"
#include "net/Network.h"
#include "net/Url.h"
#include "Options.h"
#include "workers/Workers.h"


Network::Network(const Options *options) :
    m_donate(false),
    m_options(options),
    m_pool(0),
    m_diff(0)
{
    Workers::setListener(this);

    m_pools.reserve(2);
    m_agent = userAgent();

    addPool(std::make_unique<Url>().get());
//    addPool(m_options->url());
//    addPool(m_options->backupUrl());

    m_timer.data = this;
    uv_timer_init(uv_default_loop(), &m_timer);
}


Network::~Network()
{
    for (auto client : m_pools) {
        delete client;
    }

    free(m_agent);
}


void Network::connect()
{
    m_pools[1]->connect();

    if (m_options->donateLevel()) {
        uv_timer_start(&m_timer, Network::onTimer, (100 - m_options->donateLevel()) * 60 * 1000, 0);
    }
}


void Network::onClose(Client *client, int failures)
{
    const int id = client->id();
    if (id == 0) {
        if (failures == -1) {
            stopDonate();
        }

        return;
    }

    if (m_pool == id) {
        m_pool = 0;
        Workers::pause();
    }

    if (id == 1 && m_pools.size() > 2 && failures == m_options->retries()) {
        m_pools[2]->connect();
    }
}


void Network::onJobReceived(Client *client, const Job &job)
{
    if (m_donate && client->id() != 0) {
        return;
    }

    setJob(client, job);
}


void Network::onJobResult(const JobResult &result)
{
    if (m_options->colors()) {
        LOG_NOTICE("\x1B[01;32mSHARE FOUND");
    }
    else {
        LOG_NOTICE("SHARE FOUND");
    }

    m_pools[result.poolId]->submit(result);
}


void Network::onLoginCredentialsRequired(Client *client)
{
//    client->login(m_options->user(), m_options->pass(), m_agent);
}


void Network::onLoginSuccess(Client *client)
{
    const int id = client->id();
    if (id == 0) {
        return startDonate();
    }

    if (id == 2 && m_pool) { // primary pool is already active
        m_pools[2]->disconnect();
        return;
    }

    LOG_INFO(m_options->colors() ? "\x1B[01;37muse pool: \x1B[01;36m%s:%d" : "use pool: %s:%d", client->host(), client->port());
    m_pool = id;

    if (m_pool == 1 && m_pools.size() > 2) { // try disconnect from backup pool
        m_pools[2]->disconnect();
    }
}


void Network::addPool(const Url *url)
{
    if (!url) {
        return;
    }

    Client *client = new Client(m_pools.size(), this);
    client->setUrl(url);
    client->setRetryPause(m_options->retryPause() * 1000);
    client->setKeepAlive(m_options->keepAlive());

    m_pools.push_back(client);
}


void Network::setJob(Client *client, const Job &job)
{
    if (m_options->colors()) {
        LOG_INFO("\x1B[01;35mnew job\x1B[0m from \x1B[01;37m%s:%d\x1B[0m diff: \x1B[01;37m%d", client->host(), client->port(), job.diff());

    }
    else {
        LOG_INFO("new job from %s:%d diff: %d", client->host(), client->port(), job.diff());
    }

    Workers::setJob(job);
}


void Network::startDonate()
{
    if (m_donate) {
        return;
    }

    LOG_NOTICE("dev donate started");

    m_donate = true;
}


void Network::stopDonate()
{
    if (!m_donate) {
        return;
    }

    LOG_NOTICE("dev donate finished");

    m_donate = false;
    if (!m_pool) {
        return;
    }

    Client *client = m_pools[m_pool];
    if (client->isReady()) {
        setJob(client, client->job());
    }
}


void Network::onTimer(uv_timer_t *handle)
{
    auto net = static_cast<Network*>(handle->data);

    if (!net->m_donate) {
        auto url = std::make_unique<Url>("donate.xmrig.com", net->m_options->algo() == Options::ALGO_CRYPTONIGHT_LITE ? 3333 : 443);
        net->m_pools[0]->connect(url.get());

        uv_timer_start(&net->m_timer, Network::onTimer, net->m_options->donateLevel() * 60 * 1000, 0);
        return;
    }

    net->m_pools[0]->disconnect();
    uv_timer_start(&net->m_timer, Network::onTimer, (100 - net->m_options->donateLevel()) * 60 * 1000, 0);
}
