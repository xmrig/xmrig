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


#include <uv.h>
#include <memory>


#include "Console.h"
#include "net/Client.h"
#include "net/Network.h"
#include "net/Url.h"
#include "Options.h"


Network::Network(const Options *options) :
    m_donate(false),
    m_options(options),
    m_pool(1)
{
    m_pools.reserve(2);
    m_agent = userAgent();

    std::unique_ptr<Url> url(new Url("donate.xmrig.com", 443));

    addPool(url.get());
    addPool(m_options->url());
    addPool(m_options->backupUrl());
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
    m_pools.at(m_pool)->connect();
}


void Network::onClose(Client *client, int failures)
{
    LOG_DEBUG("CLOSE %d %d", client->id(), failures);
}


void Network::onJobReceived(Client *client, const Job &job)
{

}


void Network::onLoginCredentialsRequired(Client *client)
{
    client->login(m_options->user(), m_options->pass(), m_agent);
}


void Network::onLoginSuccess(Client *client)
{
}


void Network::addPool(const Url *url)
{
    if (!url) {
        return;
    }

    Client *client = new Client(m_pools.size(), this);
    client->setUrl(url);
    client->setRetryPause(m_options->retryPause() * 1000);

    m_pools.push_back(client);
}
