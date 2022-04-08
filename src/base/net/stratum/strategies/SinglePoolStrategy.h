/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_SINGLEPOOLSTRATEGY_H
#define XMRIG_SINGLEPOOLSTRATEGY_H


#include "base/kernel/interfaces/IClientListener.h"
#include "base/kernel/interfaces/IStrategy.h"
#include "base/tools/Object.h"


namespace xmrig {


class Client;
class IStrategyListener;
class Pool;


class SinglePoolStrategy : public IStrategy, public IClientListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(SinglePoolStrategy)

    SinglePoolStrategy(const Pool &pool, int retryPause, int retries, IStrategyListener *listener, bool quiet = false);
    ~SinglePoolStrategy() override;

protected:
    inline bool isActive() const override           { return m_active; }
    inline IClient *client() const override         { return m_client; }

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
    bool m_active;
    IClient *m_client;
    IStrategyListener *m_listener;
};


} /* namespace xmrig */


#endif /* XMRIG_SINGLEPOOLSTRATEGY_H */
