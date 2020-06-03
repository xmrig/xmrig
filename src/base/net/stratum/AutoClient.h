/* XMRig
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

#ifndef XMRIG_AUTOCLIENT_H
#define XMRIG_AUTOCLIENT_H


#include "base/net/stratum/EthStratumClient.h"


#include <utility>


namespace xmrig {


class AutoClient : public EthStratumClient
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(AutoClient)

    AutoClient(int id, const char *agent, IClientListener *listener);
    ~AutoClient() override = default;

protected:
    inline void login() override    { Client::login(); }

    bool handleResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error) override;
    bool parseLogin(const rapidjson::Value &result, int *code) override;
    int64_t submit(const JobResult &result) override;
    void parseNotification(const char *method, const rapidjson::Value &params, const rapidjson::Value &error) override;

private:
    enum Mode {
        DEFAULT_MODE,
        ETH_MODE
    };

    Mode m_mode = DEFAULT_MODE;
};


} /* namespace xmrig */


#endif /* XMRIG_AUTOCLIENT_H */
