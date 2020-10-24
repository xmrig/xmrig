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
#include "base/net/http/HttpListener.h"
#include "base/net/stratum/benchmark/BenchConfig.h"


xmrig::BenchClient::BenchClient(const std::shared_ptr<BenchConfig> &benchmark, IClientListener* listener) :
    m_listener(listener),
    m_benchmark(benchmark)
{
    m_httpListener = std::make_shared<HttpListener>(this);

    std::vector<char> blob(112 * 2 + 1, '0');
    blob.back() = '\0';

    m_job.setBlob(blob.data());
    m_job.setAlgorithm(m_benchmark->algorithm());
    m_job.setDiff(std::numeric_limits<uint64_t>::max());
    m_job.setHeight(1);
    m_job.setBenchSize(m_benchmark->size());
    m_job.setBenchHash(m_benchmark->hash());

    if (m_benchmark->isSubmit()) {
        m_mode = ONLINE_BENCH;

        return;
    }

    if (!m_benchmark->id().isEmpty()) {
        m_job.setId(m_benchmark->id());
        m_mode = ONLINE_VERIFY;

        return;
    }

    m_job.setId("00000000");

    if (m_job.benchHash() && m_job.setSeedHash(m_benchmark->seed())) {
        m_mode = STATIC_VERIFY;

        return;
    }

    blob[Job::kMaxSeedSize * 2] = '\0';
    m_job.setSeedHash(blob.data());
}


void xmrig::BenchClient::connect()
{
    if (m_mode == STATIC_BENCH || m_mode == STATIC_VERIFY) {
        m_listener->onLoginSuccess(this);
        m_listener->onJobReceived(this, m_job, rapidjson::Value());
    }
}


void xmrig::BenchClient::setPool(const Pool &pool)
{
    m_pool = pool;
}


void xmrig::BenchClient::onHttpData(const HttpData &data)
{
}
