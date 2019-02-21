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

#ifndef XMRIG_FAILOVERSTRATEGY_H
#define XMRIG_FAILOVERSTRATEGY_H


#include <vector>


#include "base/net/Pool.h"
#include "common/interfaces/IClientListener.h"
#include "common/interfaces/IStrategy.h"


namespace xmrig {


class Client;
class IStrategyListener;


class FailoverStrategy : public IStrategy, public IClientListener
{
public:
    FailoverStrategy(const std::vector<Pool> &pool, int retryPause, int retries, IStrategyListener *listener, bool quiet = false);
    FailoverStrategy(int retryPause, int retries, IStrategyListener *listener, bool quiet = false);
    ~FailoverStrategy() override;

    void add(const Pool &pool);

public:
    inline bool isActive() const override  { return m_active >= 0; }

    int64_t submit(const JobResult &result) override;
    void connect() override;
    void resume() override;
    void setAlgo(const Algorithm &algo) override;
    void stop() override;
    void tick(uint64_t now) override;

protected:
    void onClose(Client *client, int failures) override;
    void onJobReceived(Client *client, const Job &job) override;
    void onLoginSuccess(Client *client) override;
    void onResultAccepted(Client *client, const SubmitResult &result, const char *error) override;

private:
    inline Client *active() const { return m_pools[static_cast<size_t>(m_active)]; }

    const bool m_quiet;
    const int m_retries;
    const int m_retryPause;
    int m_active;
    int m_index;
    IStrategyListener *m_listener;
    std::vector<Client*> m_pools;
};


} /* namespace xmrig */

#endif /* XMRIG_FAILOVERSTRATEGY_H */
