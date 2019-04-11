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


#include "base/kernel/interfaces/IStrategyListener.h"
#include "base/net/stratum/Client.h"
#include "base/net/stratum/strategies/SinglePoolStrategy.h"
#include "common/Platform.h"


#ifdef XMRIG_FEATURE_HTTP
#   include "base/net/stratum/DaemonClient.h"
#endif


xmrig::SinglePoolStrategy::SinglePoolStrategy(const Pool &pool, int retryPause, int retries, IStrategyListener *listener, bool quiet) :
    m_active(false),
    m_listener(listener)
{
#   ifdef XMRIG_FEATURE_HTTP
    if (!pool.isDaemon()) {
        m_client = new Client(0, Platform::userAgent(), this);
    }
    else {
        m_client = new DaemonClient(0, this);
    }
#   else
    m_client = new Client(0, Platform::userAgent(), this);
#   endif

    m_client->setPool(pool);
    m_client->setRetries(retries);
    m_client->setRetryPause(retryPause * 1000);
    m_client->setQuiet(quiet);
}


xmrig::SinglePoolStrategy::~SinglePoolStrategy()
{
    m_client->deleteLater();
}


int64_t xmrig::SinglePoolStrategy::submit(const JobResult &result)
{
    return m_client->submit(result);
}


void xmrig::SinglePoolStrategy::connect()
{
    m_client->connect();
}


void xmrig::SinglePoolStrategy::resume()
{
    if (!isActive()) {
        return;
    }

    m_listener->onJob(this, m_client, m_client->job());
}


void xmrig::SinglePoolStrategy::setAlgo(const xmrig::Algorithm &algo)
{
    m_client->setAlgo(algo);
}


void xmrig::SinglePoolStrategy::stop()
{
    m_client->disconnect();
}


void xmrig::SinglePoolStrategy::tick(uint64_t now)
{
    m_client->tick(now);
}


void xmrig::SinglePoolStrategy::onClose(IClient *, int)
{
    if (!isActive()) {
        return;
    }

    m_active = false;
    m_listener->onPause(this);
}


void xmrig::SinglePoolStrategy::onJobReceived(IClient *client, const Job &job, const rapidjson::Value &)
{
    m_listener->onJob(this, client, job);
}


void xmrig::SinglePoolStrategy::onLoginSuccess(IClient *client)
{
    m_active = true;
    m_listener->onActive(this, client);
}


void xmrig::SinglePoolStrategy::onResultAccepted(IClient *client, const SubmitResult &result, const char *error)
{
    m_listener->onResultAccepted(this, client, result, error);
}
