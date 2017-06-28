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


#include "net/Client.h"
#include "net/strategies/DonateStrategy.h"
#include "Options.h"


DonateStrategy::DonateStrategy(const char *agent, IStrategyListener *listener) :
    m_listener(listener)
{
    Url *url = new Url("donate.xmrig.com", Options::i()->algo() == Options::ALGO_CRYPTONIGHT_LITE ? 3333 : 443, Options::i()->pools().front()->user());

    m_client = new Client(-1, agent, this);
    m_client->setUrl(url);
    m_client->setRetryPause(Options::i()->retryPause() * 1000);

    delete url;

    m_timer.data = this;
    uv_timer_init(uv_default_loop(), &m_timer);

    uv_timer_start(&m_timer, DonateStrategy::onTimer, (100 - Options::i()->donateLevel()) * 60 * 1000, 0);
}


void DonateStrategy::connect()
{
}


void DonateStrategy::onClose(Client *client, int failures)
{

}


void DonateStrategy::onJobReceived(Client *client, const Job &job)
{

}


void DonateStrategy::onLoginSuccess(Client *client)
{
}


void DonateStrategy::onTimer(uv_timer_t *handle)
{

}
