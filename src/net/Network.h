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

#ifndef XMRIG_NETWORK_H
#define XMRIG_NETWORK_H


#include <vector>
#include <uv.h>


#include "api/NetworkState.h"
#include "common/interfaces/IControllerListener.h"
#include "common/interfaces/IStrategyListener.h"
#include "interfaces/IJobResultListener.h"


namespace xmrig {


class Controller;
class IStrategy;


class Network : public IJobResultListener, public IStrategyListener, public IControllerListener
{
public:
    Network(Controller *controller);
    ~Network() override;

    void connect();
    void stop();

protected:
    void onActive(IStrategy *strategy, Client *client) override;
    void onConfigChanged(Config *config, Config *previousConfig) override;
    void onJob(IStrategy *strategy, Client *client, const Job &job) override;
    void onJobResult(const JobResult &result) override;
    void onPause(IStrategy *strategy) override;
    void onResultAccepted(IStrategy *strategy, Client *client, const SubmitResult &result, const char *error) override;

private:
    constexpr static int kTickInterval = 1 * 1000;

    bool isColors() const;
    void setJob(Client *client, const Job &job, bool donate);
    void tick();

    static void onTick(uv_timer_t *handle);

    IStrategy *m_donate;
    IStrategy *m_strategy;
    NetworkState m_state;
    uv_timer_t m_timer;
};


} /* namespace xmrig */


#endif /* XMRIG_NETWORK_H */
