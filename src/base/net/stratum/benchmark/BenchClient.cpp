/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "base/net/stratum/benchmark/BenchClient.h"
#include "3rdparty/rapidjson/document.h"
#include "base/kernel/interfaces/IClientListener.h"
#include "base/net/stratum/benchmark/BenchConfig.h"


xmrig::BenchClient::BenchClient(const std::shared_ptr<BenchConfig> &benchmark, IClientListener* listener) :
    m_listener(listener)
{
    m_job.setAlgorithm(benchmark->algorithm());

    std::vector<char> blob(112 * 2 + 1, '0');

    blob.back() = '\0';
    m_job.setBlob(blob.data());

    blob[Job::kMaxSeedSize * 2] = '\0';
    m_job.setSeedHash(blob.data());

    m_job.setDiff(uint64_t(-1));
    m_job.setHeight(1);

    m_job.setId("00000000");
}


void xmrig::BenchClient::connect()
{
    m_listener->onLoginSuccess(this);

    rapidjson::Value params;
    m_listener->onJobReceived(this, m_job, params);
}


void xmrig::BenchClient::setPool(const Pool &pool)
{
    m_pool = pool;
}
