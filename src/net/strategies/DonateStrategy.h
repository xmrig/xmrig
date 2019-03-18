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

#ifndef XMRIG_DONATESTRATEGY_H
#define XMRIG_DONATESTRATEGY_H


#include <vector>


#include "base/kernel/interfaces/IClientListener.h"
#include "base/kernel/interfaces/IStrategy.h"
#include "base/kernel/interfaces/IStrategyListener.h"
#include "base/kernel/interfaces/ITimerListener.h"
#include "base/net/stratum/Pool.h"


namespace xmrig {


class Client;
class Controller;
class IStrategyListener;


class DonateStrategy : public IStrategy, public IStrategyListener, public ITimerListener
{
public:
    DonateStrategy(Controller *controller, IStrategyListener *listener);
    ~DonateStrategy() override;

public:
    inline bool isActive() const override  { return state() == STATE_ACTIVE; }
    inline void resume() override          {}

    int64_t submit(const JobResult &result) override;
    void connect() override;
    void setAlgo(const Algorithm &algo) override;
    void stop() override;
    void tick(uint64_t now) override;

protected:
    void onActive(IStrategy *strategy, Client *client) override;
    void onJob(IStrategy *strategy, Client *client, const Job &job) override;
    void onPause(IStrategy *strategy) override;
    void onResultAccepted(IStrategy *strategy, Client *client, const SubmitResult &result, const char *error) override;
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

    void idle(double min, double max);
    void setState(State state);

    Client *m_proxy;
    const uint64_t m_donateTime;
    const uint64_t m_idleTime;
    Controller *m_controller;
    IStrategy *m_strategy;
    IStrategyListener *m_listener;
    State m_state;
    std::vector<Pool> m_pools;
    Timer *m_timer;
    uint64_t m_now;
    uint64_t m_stop;
};


} /* namespace xmrig */


#endif /* XMRIG_DONATESTRATEGY_H */
