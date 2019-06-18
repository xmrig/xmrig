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


#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <uv.h>


#include "base/kernel/interfaces/IClient.h"
#include "base/net/stratum/Pool.h"
#include "base/net/stratum/SubmitResult.h"
#include "base/tools/Chrono.h"
#include "net/NetworkState.h"


xmrig::NetworkState::NetworkState() :
    pool(),
    accepted(0),
    diff(0),
    failures(0),
    rejected(0),
    total(0),
    m_active(false)
{
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
        rejected++;
        return;
    }

    accepted++;
    total += result.diff;

    const size_t ln = topDiff.size() - 1;
    if (result.actualDiff > topDiff[ln]) {
        topDiff[ln] = result.actualDiff;
        std::sort(topDiff.rbegin(), topDiff.rend());
    }

    m_latency.push_back(result.elapsed > 0xFFFF ? 0xFFFF : static_cast<uint16_t>(result.elapsed));
}


void xmrig::NetworkState::onActive(IClient *client)
{
    snprintf(pool, sizeof(pool) - 1, "%s:%d", client->pool().host().data(), client->pool().port());

    m_ip             = client->ip();
    m_tls            = client->tlsVersion();
    m_fingerprint    = client->tlsFingerprint();
    m_active         = true;
    m_connectionTime = Chrono::steadyMSecs();
}


void xmrig::NetworkState::stop()
{
    m_active      = false;
    diff          = 0;
    m_ip          = nullptr;
    m_tls         = nullptr;
    m_fingerprint = nullptr;

    failures++;
    m_latency.clear();
}
