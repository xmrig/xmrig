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


#include "Console.h"
#include "net/Client.h"
#include "net/Network.h"
#include "Options.h"


Network::Network(const Options *options) :
    m_backupPool(nullptr),
    m_donatePool(nullptr),
    m_pool(nullptr),
    m_options(options)
{
    m_agent = userAgent();
    m_pool = new Client(this);
}


Network::~Network()
{
    delete m_pool;
    delete m_donatePool;
    delete m_backupPool;

    free(m_agent);
}


void Network::connect()
{
    m_pool->connect(m_options->url());
//    LOG_DEBUG("XX %s", m_options->url());
}


void Network::onLoginCredentialsRequired(Client *client)
{
    client->login(m_options->user(), m_options->pass(), m_agent);
}
