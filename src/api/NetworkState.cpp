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


#include <algorithm>
#include <string.h>
#include <uv.h>


#include "api/NetworkState.h"
#include "net/SubmitResult.h"


NetworkState::NetworkState() :
    diff(0),
    accepted(0),
    failures(0),
    rejected(0),
    total(0),
    m_active(false)
{
    memset(pool, 0, sizeof(pool));
}


int NetworkState::connectionTime() const
{
    return m_active ? ((uv_now(uv_default_loop()) - m_connectionTime) / 1000) : 0;
}


uint32_t NetworkState::avgTime() const
{
    if (m_latency.empty()) {
        return 0;
    }

    return (uint32_t) connectionTime() / m_latency.size();
}


uint32_t NetworkState::latency() const
{
    const size_t calls = m_latency.size();
    if (calls == 0) {
        return 0;
    }

    auto v = m_latency;
    std::nth_element(v.begin(), v.begin() + calls / 2, v.end());

    return v[calls / 2];
}


void NetworkState::add(const SubmitResult &result, const char *error)
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

    m_latency.push_back(result.elapsed > 0xFFFF ? 0xFFFF : (uint16_t) result.elapsed);
}


void NetworkState::setPool(const char *host, int port, const char *ip)
{
    snprintf(pool, sizeof(pool) - 1, "%s:%d", host, port);

    m_active = true;
    m_connectionTime = uv_now(uv_default_loop());
}


void NetworkState::stop()
{
    m_active = false;
    diff     = 0;

    failures++;
    m_latency.clear();
}
