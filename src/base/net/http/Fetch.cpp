/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/net/http/Fetch.h"
#include "3rdparty/rapidjson/document.h"
#include "3rdparty/rapidjson/stringbuffer.h"
#include "3rdparty/rapidjson/writer.h"
#include "base/io/log/Log.h"
#include "base/net/http/HttpClient.h"


#ifdef XMRIG_FEATURE_TLS
#   include "base/net/https/HttpsClient.h"
#endif


xmrig::FetchRequest::FetchRequest(http_method method, const String &host, uint16_t port, const String &path, bool tls, bool quiet, const char *data, size_t size, const char *contentType) :
    quiet(quiet),
    tls(tls),
    method(method),
    host(host),
    path(path),
    port(port)
{
    assert(port > 0);

    setBody(data, size, contentType);
}


xmrig::FetchRequest::FetchRequest(http_method method, const String &host, uint16_t port, const String &path, const rapidjson::Value &value, bool tls, bool quiet) :
    quiet(quiet),
    tls(tls),
    method(method),
    host(host),
    path(path),
    port(port)
{
    assert(port > 0);

    setBody(value);
}


void xmrig::FetchRequest::setBody(const char *data, size_t size, const char *contentType)
{
    if (!data) {
        return;
    }

    assert(method != HTTP_GET && method != HTTP_HEAD);

    if (method == HTTP_GET || method == HTTP_HEAD) {
        return;
    }

    body = size ? std::string(data, size) : data;
    if (contentType) {
        headers.insert({ HttpData::kContentType, contentType });
    }
}


void xmrig::FetchRequest::setBody(const rapidjson::Value &value)
{
    assert(method != HTTP_GET && method != HTTP_HEAD);

    if (method == HTTP_GET || method == HTTP_HEAD) {
        return;
    }

    using namespace rapidjson;

    StringBuffer buffer(nullptr, 512);
    Writer<StringBuffer> writer(buffer);
    value.Accept(writer);

    setBody(buffer.GetString(), buffer.GetSize(), HttpData::kApplicationJson.c_str());
}


void xmrig::fetch(const char *tag, FetchRequest &&req, const std::weak_ptr<IHttpListener> &listener, int type, uint64_t rpcId)
{
#   ifdef APP_DEBUG
    LOG_DEBUG(CYAN("http%s://%s:%u ") MAGENTA_BOLD("\"%s %s\"") BLACK_BOLD(" body: ") CYAN_BOLD("%zu") BLACK_BOLD(" bytes"),
              req.tls ? "s" : "", req.host.data(), req.port, http_method_str(req.method), req.path.data(), req.body.size());

    if (req.hasBody() && req.body.size() < (Log::kMaxBufferSize - 1024) && req.headers.count(HttpData::kContentType) && req.headers.at(HttpData::kContentType) == HttpData::kApplicationJson) {
        Log::print(BLUE_BG_BOLD("%s:") BLACK_BOLD_S " %.*s", req.headers.at(HttpData::kContentType).c_str(), static_cast<int>(req.body.size()), req.body.c_str());
    }
#   endif

    HttpClient *client;
#   ifdef XMRIG_FEATURE_TLS
    if (req.tls) {
        client = new HttpsClient(tag, std::move(req), listener);
    }
    else
#   endif
    {
        client = new HttpClient(tag, std::move(req), listener);
    }

    client->userType = type;
    client->rpcId    = rpcId;
    client->connect();
}
