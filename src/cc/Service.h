/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2017-     BenDr0id    <ben@graef.in>
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

#ifndef __SERVICE_H__
#define __SERVICE_H__

#define CONTENT_TYPE_HTML "text/html"
#define CONTENT_TYPE_JSON "application/json"

#include <string>
#include <uv.h>
#include <microhttpd.h>
#include <map>
#include "Options.h"
#include "ClientStatus.h"
#include "ControlCommand.h"

class Service
{
public:
    static bool start();
    static void release();

    static unsigned handleGET(const Options* options, const std::string& url, const std::string& clientId, std::string& resp);
    static unsigned handlePOST(const Options* options, const std::string& url, const std::string& clientIp, const std::string& clientId, const std::string& data, std::string& resp);

private:
    static unsigned getClientConfig(const Options* options, const std::string& clientId, std::string& resp);
    static unsigned getClientCommand(const std::string& clientId, std::string& resp);
    static unsigned getClientStatusList(std::string& resp);
    static unsigned getAdminPage(const Options* options, std::string& resp);

    static unsigned setClientStatus(const std::string& clientIp, const std::string& clientId, const std::string& data, std::string& resp);
    static unsigned setClientCommand(const std::string& clientId, const std::string& data, std::string& resp);
    static unsigned setClientConfig(const Options* options, const std::string &clientId, const std::string &data, std::string &resp);
    static unsigned resetClientStatusList(const std::string& data, std::string& resp);

    static std::string getClientConfigFileName(const Options *options, const std::string &clientId);

private:
    static int m_currentServerTime;

    static std::map<std::string, ClientStatus> m_clientStatus;
    static std::map<std::string, ControlCommand> m_clientCommand;

    static uv_mutex_t m_mutex;

};

#endif /* __SERVICE_H__ */
