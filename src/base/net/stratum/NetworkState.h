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

#ifndef XMRIG_NETWORKSTATE_H
#define XMRIG_NETWORKSTATE_H


#include "base/crypto/Algorithm.h"
#include "base/net/stratum/strategies/StrategyProxy.h"
#include "base/tools/String.h"


#include <array>
#include <vector>


namespace xmrig {


class NetworkState : public StrategyProxy
{
public:
    NetworkState(IStrategyListener *listener);

    inline const Algorithm &algorithm() const   { return m_algorithm; }
    inline uint64_t accepted() const            { return m_accepted; }
    inline uint64_t rejected() const            { return m_rejected; }

#   ifdef XMRIG_FEATURE_API
    rapidjson::Value getConnection(rapidjson::Document &doc, int version) const;
    rapidjson::Value getResults(rapidjson::Document &doc, int version) const;
#   endif

protected:
    void onActive(IStrategy *strategy, IClient *client) override;
    void onJob(IStrategy *strategy, IClient *client, const Job &job, const rapidjson::Value &params) override;
    void onPause(IStrategy *strategy) override;
    void onResultAccepted(IStrategy *strategy, IClient *client, const SubmitResult &result, const char *error) override;

private:
    uint32_t avgTime() const;
    uint32_t latency() const;
    uint64_t connectionTime() const;
    void add(const SubmitResult &result, const char *error);
    void stop();

    Algorithm m_algorithm;
    bool m_active               = false;
    char m_pool[256]{};
    std::array<uint64_t, 10> topDiff { { } };
    std::vector<uint16_t> m_latency;
    String m_fingerprint;
    String m_ip;
    String m_tls;
    uint64_t m_accepted         = 0;
    uint64_t m_connectionTime   = 0;
    uint64_t m_diff             = 0;
    uint64_t m_failures         = 0;
    uint64_t m_hashes           = 0;
    uint64_t m_rejected         = 0;
};


} /* namespace xmrig */


#endif /* XMRIG_NETWORKSTATE_H */
