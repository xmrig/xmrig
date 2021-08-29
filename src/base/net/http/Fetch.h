/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#ifndef XMRIG_FETCH_H
#define XMRIG_FETCH_H


#include "3rdparty/llhttp/llhttp.h"
#include "3rdparty/rapidjson/fwd.h"
#include "base/tools/String.h"


#include <map>
#include <memory>
#include <string>


namespace xmrig {


class IHttpListener;


class FetchRequest
{
public:
    FetchRequest() = default;
    FetchRequest(llhttp_method method, const String &host, uint16_t port, const String &path, bool tls = false, bool quiet = false, const char *data = nullptr, size_t size = 0, const char *contentType = nullptr);
    FetchRequest(llhttp_method method, const String &host, uint16_t port, const String &path, const rapidjson::Value &value, bool tls = false, bool quiet = false);

    void setBody(const char *data, size_t size, const char *contentType = nullptr);
    void setBody(const rapidjson::Value &value);

    inline bool hasBody() const { return method != HTTP_GET && method != HTTP_HEAD && !body.empty(); }

    bool quiet              = false;
    bool tls                = false;
    llhttp_method method    = HTTP_GET;
    std::map<const std::string, const std::string> headers;
    std::string body;
    String fingerprint;
    String host;
    String path;
    uint16_t port           = 0;
    uint64_t timeout        = 0;
};


void fetch(const char *tag, FetchRequest &&req, const std::weak_ptr<IHttpListener> &listener, int type = 0, uint64_t rpcId = 0);


} // namespace xmrig


#endif // XMRIG_FETCH_H

