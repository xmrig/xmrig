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


#include "interfaces/IStrategyListener.h"
#include "net/Client.h"
#include "net/strategies/FailoverStrategy.h"
#include "Options.h"


FailoverStrategy::FailoverStrategy(const std::vector<Url*> &urls, const char *agent, IStrategyListener *listener) :
    m_active(-1),
    m_index(0),
    m_listener(listener)
{
    for (const Url *url : urls) {
        add(url, agent);
    }
}


int64_t FailoverStrategy::submit(const JobResult &result)
{
    return m_pools[m_active]->submit(result);
}


void FailoverStrategy::connect()
{
    m_pools[m_index]->connect();
}


void FailoverStrategy::resume()
{
    if (!isActive()) {
        return;
    }

    m_listener->onJob( m_pools[m_active],  m_pools[m_active]->job());
}


void FailoverStrategy::stop()
{
    for (size_t i = 0; i < m_pools.size(); ++i) {
        m_pools[i]->disconnect();
    }

    m_index  = 0;
    m_active = -1;

    m_listener->onPause(this);
}


void FailoverStrategy::tick(uint64_t now)
{
    for (Client *client : m_pools) {
        client->tick(now);
    }
}


void FailoverStrategy::onClose(Client *client, int failures)
{
    if (failures == -1) {
        return;
    }

    if (m_active == client->id()) {
        m_active = -1;
        m_listener->onPause(this);
    }

    if (m_index == 0 && failures < Options::i()->retries()) {
        return;
    }

    if (m_index == client->id() && (m_pools.size() - m_index) > 1) {
        m_pools[++m_index]->connect();
    }
}


void FailoverStrategy::onJobReceived(Client *client, const Job &job)
{
    if (m_active == client->id()) {
        m_listener->onJob(client, job);
    }
}


void FailoverStrategy::onLoginSuccess(Client *client)
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
        m_listener->onActive(client);
    }
}


void FailoverStrategy::onResultAccepted(Client *client, const SubmitResult &result, const char *error)
{
    m_listener->onResultAccepted(client, result, error);
}


void FailoverStrategy::add(const Url *url, const char *agent)
{
    Client *client = new Client((int) m_pools.size(), agent, this);
    client->setUrl(url);
    client->setRetryPause(Options::i()->retryPause() * 1000);

    m_pools.push_back(client);
}
