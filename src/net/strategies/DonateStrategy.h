/* XMRig
 * Copyright (c) 2018-2023 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2023 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_DONATESTRATEGY_H
#define XMRIG_DONATESTRATEGY_H


#include "base/kernel/interfaces/IClientListener.h"
#include "base/kernel/interfaces/IStrategy.h"
#include "base/kernel/interfaces/IStrategyListener.h"
#include "base/kernel/interfaces/ITimerListener.h"
#include "base/net/stratum/Pool.h"
#include "base/tools/Buffer.h"


namespace xmrig {


class Client;
class Controller;


class DonateStrategy : public IStrategy, public IStrategyListener, public ITimerListener, public IClientListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(DonateStrategy)

    DonateStrategy(Controller *controller, IStrategyListener *listener);
    ~DonateStrategy() override;

    void update(IClient *client, const Job &job);

protected:
    inline bool isActive() const override                                                                              { return state() == STATE_ACTIVE; }
    inline IClient *client() const override                                                                            { return m_proxy ? m_proxy.get() : m_strategy->client(); }
    inline void onJob(IStrategy *, IClient *client, const Job &job, const rapidjson::Value &params) override           { setJob(client, job, params); }
    inline void onJobReceived(IClient *client, const Job &job, const rapidjson::Value &params) override                { setJob(client, job, params); }
    inline void onResultAccepted(IClient *client, const SubmitResult &result, const char *error) override              { setResult(client, result, error); }
    inline void onResultAccepted(IStrategy *, IClient *client, const SubmitResult &result, const char *error) override { setResult(client, result, error); }
    inline void resume() override                                                                                      {}

    int64_t submit(const JobResult &result) override;
    void connect() override;
    void setAlgo(const Algorithm &algo) override;
    void setProxy(const ProxyUrl &proxy) override;
    void stop() override;
    void tick(uint64_t now) override;

    void onActive(IStrategy *strategy, IClient *client) override;
    void onPause(IStrategy *strategy) override;

    void onClose(IClient *client, int failures) override;
    void onLogin(IClient *client, rapidjson::Document &doc, rapidjson::Value &params) override;
    void onLogin(IStrategy *strategy, IClient *client, rapidjson::Document &doc, rapidjson::Value &params) override;
    void onLoginSuccess(IClient *client) override;
    void onVerifyAlgorithm(const IClient *client, const Algorithm &algorithm, bool *ok) override;
    void onVerifyAlgorithm(IStrategy *strategy, const IClient *client, const Algorithm &algorithm, bool *ok) override;

    void onTimer(const Timer *timer) override;

private:
    enum State {
        STATE_NEW,
        STATE_IDLE,
        STATE_CONNECT,
        STATE_ACTIVE,
        STATE_WAIT
    };

    inline State state() const { return m_state; }

    std::shared_ptr<IClient> createProxy();
    void idle(double min, double max);
    void setJob(IClient *client, const Job &job, const rapidjson::Value &params);
    void setParams(rapidjson::Document &doc, rapidjson::Value &params);
    void setResult(IClient *client, const SubmitResult &result, const char *error);
    void setState(State state);

    Algorithm m_algorithm;
    bool m_tls                      = false;
    Buffer m_seed;
    char m_userId[65]               = { 0 };
    const uint64_t m_donateTime;
    const uint64_t m_idleTime;
    Controller *m_controller;
    std::shared_ptr<IClient> m_proxy;
    std::shared_ptr<IStrategy> m_strategy;
    IStrategyListener *m_listener;
    State m_state                   = STATE_NEW;
    std::vector<Pool> m_pools;
    std::shared_ptr<Timer> m_timer;
    uint64_t m_diff                 = 0;
    uint64_t m_height               = 0;
    uint64_t m_now                  = 0;
    uint64_t m_timestamp            = 0;
};


} // namespace xmrig


#endif // XMRIG_DONATESTRATEGY_H
