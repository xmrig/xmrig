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

#ifndef XMRIG_STRATEGYPROXY_H
#define XMRIG_STRATEGYPROXY_H


#include "base/kernel/interfaces/IStrategyListener.h"


namespace xmrig {


class StrategyProxy : public IStrategyListener
{
public:
    inline StrategyProxy(IStrategyListener *listener) : m_listener(listener) {}

protected:
    inline void onActive(IStrategy *strategy, IClient *client) override
    {
        m_listener->onActive(strategy, client);
    }

    inline void onJob(IStrategy *strategy, IClient *client, const Job &job) override
    {
        m_listener->onJob(strategy, client, job);
    }

    inline void onLogin(IStrategy *strategy, IClient *client, rapidjson::Document &doc, rapidjson::Value &params) override
    {
        m_listener->onLogin(strategy, client, doc, params);
    }

    inline void onPause(IStrategy *strategy) override
    {
        m_listener->onPause(strategy);
    }

    inline void onResultAccepted(IStrategy *strategy, IClient *client, const SubmitResult &result, const char *error) override
    {
        m_listener->onResultAccepted(strategy, client, result, error);
    }

    inline void onVerifyAlgorithm(IStrategy *strategy, const IClient *client, const Algorithm &algorithm, bool *ok) override
    {
        m_listener->onVerifyAlgorithm(strategy, client, algorithm, ok);
    }

private:
    IStrategyListener *m_listener;
};


} /* namespace xmrig */


#endif /* XMRIG_STRATEGYPROXY_H */
