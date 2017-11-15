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

#ifndef __DONATESTRATEGY_H__
#define __DONATESTRATEGY_H__


#include <uv.h>


#include "interfaces/IClientListener.h"
#include "interfaces/IStrategy.h"


class Client;
class IStrategyListener;
class Url;


class DonateStrategy : public IStrategy, public IClientListener
{
public:
    DonateStrategy(const char *agent, IStrategyListener *listener);

public:
    inline bool isActive() const override  { return m_active; }
    inline void resume() override          {}

    int64_t submit(const JobResult &result) override;
    void connect() override;
    void stop() override;
    void tick(uint64_t now) override;

protected:
    void onClose(Client *client, int failures) override;
    void onJobReceived(Client *client, const Job &job) override;
    void onLoginSuccess(Client *client) override;
    void onResultAccepted(Client *client, const SubmitResult &result, const char *error) override;

private:
    void idle();
    void suspend();

    static void onTimer(uv_timer_t *handle);

    bool m_active;
    Client *m_client;
    const int m_donateTime;
    const int m_idleTime;
    IStrategyListener *m_listener;
    uv_timer_t m_timer;
};

#endif /* __DONATESTRATEGY_H__ */
