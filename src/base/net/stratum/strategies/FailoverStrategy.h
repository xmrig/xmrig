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

#ifndef XMRIG_FAILOVERSTRATEGY_H
#define XMRIG_FAILOVERSTRATEGY_H


#include <vector>


#include "base/kernel/interfaces/IClientListener.h"
#include "base/kernel/interfaces/IStrategy.h"
#include "base/net/stratum/Pool.h"
#include "base/tools/Object.h"


namespace xmrig {


class Client;
class IStrategyListener;


class FailoverStrategy : public IStrategy, public IClientListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(FailoverStrategy)

    FailoverStrategy(const std::vector<Pool> &pool, int retryPause, int retries, IStrategyListener *listener, bool quiet = false);
    FailoverStrategy(int retryPause, int retries, IStrategyListener *listener, bool quiet = false);
    ~FailoverStrategy() override;

    void add(const Pool &pool);

protected:
    inline bool isActive() const override           { return m_active >= 0; }
    inline IClient *client() const override         { return isActive() ? active() : m_pools[m_index]; }

    int64_t submit(const JobResult &result) override;
    void connect() override;
    void resume() override;
    void setAlgo(const Algorithm &algo) override;
    void setProxy(const ProxyUrl &proxy) override;
    void stop() override;
    void tick(uint64_t now) override;

    void onClose(IClient *client, int failures) override;
    void onJobReceived(IClient *client, const Job &job, const rapidjson::Value &params) override;
    void onLogin(IClient *client, rapidjson::Document &doc, rapidjson::Value &params) override;
    void onLoginSuccess(IClient *client) override;
    void onResultAccepted(IClient *client, const SubmitResult &result, const char *error) override;
    void onVerifyAlgorithm(const IClient *client, const Algorithm &algorithm, bool *ok) override;

private:
    inline IClient *active() const { return m_pools[static_cast<size_t>(m_active)]; }

    const bool m_quiet;
    const int m_retries;
    const int m_retryPause;
    int m_active            = -1;
    IStrategyListener *m_listener;
    size_t m_index          = 0;
    std::vector<IClient*> m_pools;
};


} /* namespace xmrig */

#endif /* XMRIG_FAILOVERSTRATEGY_H */
