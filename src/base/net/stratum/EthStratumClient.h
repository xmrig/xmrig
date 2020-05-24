/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2019      jtgrassie   <https://github.com/jtgrassie>
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

#ifndef XMRIG_ETHSTRATUMCLIENT_H
#define XMRIG_ETHSTRATUMCLIENT_H


#include "base/net/stratum/Client.h"


namespace xmrig {


class IClientListener;


class EthStratumClient : public Client
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(EthStratumClient)

    EthStratumClient(int id, const char *agent, IClientListener *listener);

    void login() override;
    void onClose() override;

protected:
    int64_t submit(const JobResult& result) override;

    bool handleResponse(int64_t id, const rapidjson::Value& result, const rapidjson::Value& error) override;
    void parseNotification(const char* method, const rapidjson::Value& params, const rapidjson::Value& error) override;

    bool disconnect() override;

private:
    void OnSubscribeResponse(const rapidjson::Value& result, bool success, uint64_t elapsed);
    void OnAuthorizeResponse(const rapidjson::Value& result, bool success, uint64_t elapsed);

    bool m_authorized = false;

    uint64_t m_target = 0;
    uint64_t m_extraNonce = 0;
};


} /* namespace xmrig */


#endif /* XMRIG_ETHSTRATUMCLIENT_H */
