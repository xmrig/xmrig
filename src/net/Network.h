/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2019      Howard Chu  <https://github.com/hyc>
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

#ifndef XMRIG_NETWORK_H
#define XMRIG_NETWORK_H


#include <vector>


#include "api/interfaces/IApiListener.h"
#include "base/kernel/interfaces/IBaseListener.h"
#include "base/kernel/interfaces/IStrategyListener.h"
#include "base/kernel/interfaces/ITimerListener.h"
#include "interfaces/IJobResultListener.h"
#include "net/NetworkState.h"
#include "rapidjson/fwd.h"


namespace xmrig {


class Controller;
class IStrategy;


class Network : public IJobResultListener, public IStrategyListener, public IBaseListener, public ITimerListener, public IApiListener
{
public:
    Network(Controller *controller);
    ~Network() override;

    inline IStrategy *strategy() const { return m_strategy; }

    void connect();

protected:
    inline void onTimer(const Timer *) override { tick(); }

    void onActive(IStrategy *strategy, IClient *client) override;
    void onConfigChanged(Config *config, Config *previousConfig) override;
    void onJob(IStrategy *strategy, IClient *client, const Job &job) override;
    void onJobResult(const JobResult &result) override;
    void onPause(IStrategy *strategy) override;
    void onRequest(IApiRequest &request) override;
    void onResultAccepted(IStrategy *strategy, IClient *client, const SubmitResult &result, const char *error) override;

private:
    constexpr static int kTickInterval = 1 * 1000;

    void setJob(IClient *client, const Job &job, bool donate);
    void tick();

#   ifdef XMRIG_FEATURE_API
    void getConnection(rapidjson::Value &reply, rapidjson::Document &doc) const;
    void getResults(rapidjson::Value &reply, rapidjson::Document &doc) const;
#   endif

    Controller *m_controller;
    IStrategy *m_donate;
    IStrategy *m_strategy;
    NetworkState m_state;
    Timer *m_timer;
};


} /* namespace xmrig */


#endif /* XMRIG_NETWORK_H */
