/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/net/stratum/NetworkState.h"
#include "3rdparty/rapidjson/document.h"
#include "base/kernel/interfaces/IClient.h"
#include "base/kernel/interfaces/IStrategy.h"
#include "base/net/stratum/Job.h"
#include "base/net/stratum/Pool.h"
#include "base/net/stratum/SubmitResult.h"
#include "base/tools/Chrono.h"


#include <algorithm>
#include <cstdio>
#include <cstring>
#include <uv.h>



xmrig::NetworkState::NetworkState(IStrategyListener *listener) : StrategyProxy(listener)
{
}


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::NetworkState::getConnection(rapidjson::Document &doc, int version) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value connection(kObjectType);
    connection.AddMember("pool",            StringRef(m_pool), allocator);
    connection.AddMember("ip",              m_ip.toJSON(), allocator);
    connection.AddMember("uptime",          connectionTime(), allocator);
    connection.AddMember("ping",            latency(), allocator);
    connection.AddMember("failures",        m_failures, allocator);
    connection.AddMember("tls",             m_tls.toJSON(), allocator);
    connection.AddMember("tls-fingerprint", m_fingerprint.toJSON(), allocator);

    connection.AddMember("algo",            m_algorithm.toJSON(), allocator);
    connection.AddMember("diff",            m_diff, allocator);
    connection.AddMember("accepted",        m_accepted, allocator);
    connection.AddMember("rejected",        m_rejected, allocator);
    connection.AddMember("avg_time",        avgTime(), allocator);
    connection.AddMember("hashes_total",    m_hashes, allocator);

    if (version == 1) {
        connection.AddMember("error_log", Value(kArrayType), allocator);
    }

    return connection;
}


rapidjson::Value xmrig::NetworkState::getResults(rapidjson::Document &doc, int version) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value results(kObjectType);

    results.AddMember("diff_current",  m_diff, allocator);
    results.AddMember("shares_good",   m_accepted, allocator);
    results.AddMember("shares_total",  m_accepted + m_rejected, allocator);
    results.AddMember("avg_time",      avgTime(), allocator);
    results.AddMember("hashes_total",  m_hashes, allocator);

    Value best(kArrayType);
    for (uint64_t i : topDiff) {
        best.PushBack(i, allocator);
    }

    results.AddMember("best", best, allocator);

    if (version == 1) {
        results.AddMember("error_log", Value(kArrayType), allocator);
    }

    return results;
}
#endif


void xmrig::NetworkState::onActive(IStrategy *strategy, IClient *client)
{
    snprintf(m_pool, sizeof(m_pool) - 1, "%s:%d", client->pool().host().data(), client->pool().port());

    m_ip             = client->ip();
    m_tls            = client->tlsVersion();
    m_fingerprint    = client->tlsFingerprint();
    m_active         = true;
    m_connectionTime = Chrono::steadyMSecs();

    StrategyProxy::onActive(strategy, client);
}


void xmrig::NetworkState::onJob(IStrategy *strategy, IClient *client, const Job &job)
{
    m_algorithm = job.algorithm();
    m_diff      = job.diff();

    StrategyProxy::onJob(strategy, client, job);
}


void xmrig::NetworkState::onPause(IStrategy *strategy)
{
    if (!strategy->isActive()) {
        stop();
    }

    StrategyProxy::onPause(strategy);
}


void xmrig::NetworkState::onResultAccepted(IStrategy *strategy, IClient *client, const SubmitResult &result, const char *error)
{
    add(result, error);

    StrategyProxy::onResultAccepted(strategy, client, result, error);
}


uint32_t xmrig::NetworkState::avgTime() const
{
    if (m_latency.empty()) {
        return 0;
    }

    return connectionTime() / (uint32_t)m_latency.size();
}


uint32_t xmrig::NetworkState::latency() const
{
    const size_t calls = m_latency.size();
    if (calls == 0) {
        return 0;
    }

    auto v = m_latency;
    std::nth_element(v.begin(), v.begin() + calls / 2, v.end());

    return v[calls / 2];
}


uint64_t xmrig::NetworkState::connectionTime() const
{
    return m_active ? ((Chrono::steadyMSecs() - m_connectionTime) / 1000) : 0;
}


void xmrig::NetworkState::add(const SubmitResult &result, const char *error)
{
    if (error) {
        m_rejected++;
        return;
    }

    m_accepted++;
    m_hashes += result.diff;

    const size_t ln = topDiff.size() - 1;
    if (result.actualDiff > topDiff[ln]) {
        topDiff[ln] = result.actualDiff;
        std::sort(topDiff.rbegin(), topDiff.rend());
    }

    m_latency.push_back(result.elapsed > 0xFFFF ? 0xFFFF : static_cast<uint16_t>(result.elapsed));
}


void xmrig::NetworkState::stop()
{
    m_active      = false;
    m_diff        = 0;
    m_ip          = nullptr;
    m_tls         = nullptr;
    m_fingerprint = nullptr;

    m_failures++;
    m_latency.clear();
}
