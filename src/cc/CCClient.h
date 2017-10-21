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

#ifndef __CC_CLIENT_H__
#define __CC_CLIENT_H__


#include <uv.h>
#include <curl/curl.h>
#include "Options.h"
#include "ClientStatus.h"

class Hashrate;
class NetworkState;

class CCClient
{
public:
    CCClient(const Options *options);
    ~CCClient();

    static void updateHashrate(const Hashrate *hashrate);
    static void updateNetworkState(const NetworkState &results);

private:

    static void publishClientStatusReport();
    static void updateConfig();
    static CURLcode performCurl(const std::string& requestUrl, const std::string& requestBuffer, const std::string& operation,
                         std::string& responseBuffer);

    static void onReport(uv_timer_t *handle);
    static int onResponse(char* data, size_t size, size_t nmemb, std::string* responseBuffer);

    constexpr static int kTickInterval = 10 * 1000;
    const Options *m_options;

    static CCClient* m_self;
    static uv_mutex_t m_mutex;

    ClientStatus m_clientStatus;

    std::string m_serverURL;
    std::string m_authorization;

    uv_timer_t m_timer;

};

#endif /* __CC_CLIENT_H__ */
