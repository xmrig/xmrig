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


#include "common/interfaces/IStrategyListener.h"
#include "common/net/Client.h"
#include "common/net/strategies/FailoverStrategy.h"
#include "common/Platform.h"


xmrig::FailoverStrategy::FailoverStrategy(const std::vector<Pool> &pools, int retryPause, int retries, IStrategyListener *listener, bool quiet) :
    m_quiet(quiet),
    m_retries(retries),
    m_retryPause(retryPause),
    m_active(-1),
    m_index(0),
    m_listener(listener)
{
    for (const Pool &pool : pools) {
        add(pool);
    }
}


xmrig::FailoverStrategy::FailoverStrategy(int retryPause, int retries, IStrategyListener *listener, bool quiet) :
    m_quiet(quiet),
    m_retries(retries),
    m_retryPause(retryPause),
    m_active(-1),
    m_index(0),
    m_listener(listener)
{
}


xmrig::FailoverStrategy::~FailoverStrategy()
{
    for (Client *client : m_pools) {
        client->deleteLater();
    }
}


void xmrig::FailoverStrategy::add(const Pool &pool)
{
    Client *client = new Client(static_cast<int>(m_pools.size()), Platform::userAgent(), this);
    client->setPool(pool);
    client->setRetries(m_retries);
    client->setRetryPause(m_retryPause * 1000);
    client->setQuiet(m_quiet);

    m_pools.push_back(client);
}


int64_t xmrig::FailoverStrategy::submit(const JobResult &result)
{
    if (m_active == -1) {
        return -1;
    }

    return active()->submit(result);
}


void xmrig::FailoverStrategy::connect()
{
    m_pools[static_cast<size_t>(m_index)]->connect();
}


void xmrig::FailoverStrategy::resume()
{
    if (!isActive()) {
        return;
    }

    m_listener->onJob(this, active(), active()->job());
}


void xmrig::FailoverStrategy::setAlgo(const xmrig::Algorithm &algo)
{
    for (Client *client : m_pools) {
        client->setAlgo(algo);
    }
}


void xmrig::FailoverStrategy::stop()
{
    for (size_t i = 0; i < m_pools.size(); ++i) {
        m_pools[i]->disconnect();
    }

    m_index  = 0;
    m_active = -1;

    m_listener->onPause(this);
}


void xmrig::FailoverStrategy::tick(uint64_t now)
{
    for (Client *client : m_pools) {
        client->tick(now);
    }
}


void xmrig::FailoverStrategy::onClose(Client *client, int failures)
{
    if (failures == -1) {
        return;
    }

    if (m_active == client->id()) {
        m_active = -1;
        m_listener->onPause(this);
    }

    if (m_index == 0 && failures < m_retries) {
        return;
    }

    if (m_index == client->id() && (m_pools.size() - static_cast<size_t>(m_index)) > 1) {
        m_pools[static_cast<size_t>(++m_index)]->connect();
    }
}


void xmrig::FailoverStrategy::onJobReceived(Client *client, const Job &job)
{
    if (m_active == client->id()) {
        m_listener->onJob(this, client, job);
    }
}


void xmrig::FailoverStrategy::onLoginSuccess(Client *client)
{
    int active = m_active;

    if (client->id() == 0 || !isActive()) {
        active = client->id();
    }

    for (size_t i = 1; i < m_pools.size(); ++i) {
        if (active != static_cast<int>(i)) {
            m_pools[i]->disconnect();
        }
    }

    if (active >= 0 && active != m_active) {
        m_index = m_active = active;
        m_listener->onActive(this, client);
    }
}


void xmrig::FailoverStrategy::onResultAccepted(Client *client, const SubmitResult &result, const char *error)
{
    m_listener->onResultAccepted(this, client, result, error);
}
