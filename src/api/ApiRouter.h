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

#ifndef XMRIG_APIROUTER_H
#define XMRIG_APIROUTER_H


#include "api/NetworkState.h"
#include "common/interfaces/IControllerListener.h"
#include "rapidjson/fwd.h"


class Hashrate;


namespace xmrig {
    class Controller;
    class HttpReply;
    class HttpRequest;
}


class ApiRouter : public xmrig::IControllerListener
{
public:
    ApiRouter(xmrig::Controller *controller);
    ~ApiRouter() override;

    void get(const xmrig::HttpRequest &req, xmrig::HttpReply &reply) const;
    void exec(const xmrig::HttpRequest &req, xmrig::HttpReply &reply);

    void tick(const xmrig::NetworkState &results);

    static rapidjson::Value normalize(double d);

protected:
    void onConfigChanged(xmrig::Config *config, xmrig::Config *previousConfig) override;

private:
    void finalize(xmrig::HttpReply &reply, rapidjson::Document &doc) const;
    void genId(const char *id);
    void getConnection(rapidjson::Document &doc) const;
    void getHashrate(rapidjson::Document &doc) const;
    void getIdentify(rapidjson::Document &doc) const;
    void getMiner(rapidjson::Document &doc) const;
    void getResults(rapidjson::Document &doc) const;
    void getThreads(rapidjson::Document &doc) const;
    void setWorkerId(const char *id);
    void updateWorkerId(const char *id, const char *previousId);

    char m_id[32];
    char m_workerId[128];
    xmrig::NetworkState m_network;
    xmrig::Controller *m_controller;
};

#endif /* XMRIG_APIROUTER_H */
