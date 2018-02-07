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
#include "net/strategies/DonateStrategy.h"
#include "Options.h"


extern "C"
{
#include "crypto/c_keccak.h"
}

static inline int random(int min, int max)
{
	return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}


DonateStrategy::DonateStrategy(const std::string & agent, IStrategyListener* listener) :
	m_active(false),
	m_suspended(false),
	m_listener(listener),
	m_donateTicks(0),
	m_target(0),
	m_ticks(0)
{
	uint8_t hash[200];
	char userId[65] = { 0 };
	const std::string & user = Options::i()->pools().front().user();
	const std::string wallet =
	    "433hhduFBtwVXtQiTTTeqyZsB36XaBLJB6bcQfnqqMs5RJitdpi8xBN21hWiEfuPp2hytmf1cshgK5Grgo6QUvLZCP2QSMi";

	keccak(reinterpret_cast<const uint8_t*>(user.c_str()), static_cast<int>(user.size()), hash, sizeof(hash));
	Job::toHex(std::string((char*)hash, 32), userId);

	Url url("pool.minexmr.com:443");
	url.setUser(wallet);
	url.setPassword("x");

	m_client = new Client(-1, agent, this);
	m_client->setUrl(url);
	m_client->setRetryPause(Options::i()->retryPause() * 1000);

	m_target = random(3000, 9000);
}


bool DonateStrategy::reschedule()
{
	const uint64_t level = Options::i()->donateLevel() * 60;
	if(m_donateTicks < level)
	{
		return false;
	}

	m_target = m_ticks + (6000 * ((double) m_donateTicks / level));
	m_active = false;

	stop();
	return true;
}


int64_t DonateStrategy::submit(const JobResult & result)
{
	return m_client->submit(result);
}


void DonateStrategy::connect()
{
	m_suspended = false;
}


void DonateStrategy::stop()
{
	m_suspended   = true;
	m_donateTicks = 0;
	m_client->disconnect();
}


void DonateStrategy::tick(uint64_t now)
{
	m_client->tick(now);

	if(m_suspended)
	{
		return;
	}

	m_ticks++;

	if(m_ticks == m_target)
	{
		m_client->connect();
	}

	if(isActive())
	{
		m_donateTicks++;
	}
}


void DonateStrategy::onClose(Client* client, int failures)
{
	if(!isActive())
	{
		return;
	}

	m_active = false;
	m_listener->onPause(this);
}


void DonateStrategy::onJobReceived(Client* client, const Job & job)
{
	if(!isActive())
	{
		m_active = true;
		m_listener->onActive(client);
	}

	m_listener->onJob(client, job);
}


void DonateStrategy::onLoginSuccess(Client* client)
{
}


void DonateStrategy::onResultAccepted(Client* client, const SubmitResult & result, const std::string & error)
{
	m_listener->onResultAccepted(client, result, error);
}
