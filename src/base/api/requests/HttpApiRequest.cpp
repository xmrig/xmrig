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


#include "3rdparty/http-parser/http_parser.h"
#include "base/api/requests/HttpApiRequest.h"
#include "base/io/json/Json.h"
#include "base/net/http/HttpData.h"
#include "rapidjson/error/en.h"


namespace xmrig {


static const char *kError  = "error";
static const char *kId     = "id";
static const char *kResult = "result";


static inline const char *rpcError(int code) {
    switch (code) {
    case IApiRequest::RPC_PARSE_ERROR:
        return "Parse error";

    case IApiRequest::RPC_INVALID_REQUEST:
        return "Invalid Request";

    case IApiRequest::RPC_METHOD_NOT_FOUND:
        return "Method not found";

    case IApiRequest::RPC_INVALID_PARAMS:
        return "Invalid params";
    }

    return "Internal error";
}


} // namespace xmrig


xmrig::HttpApiRequest::HttpApiRequest(const HttpData &req, bool restricted) :
    ApiRequest(SOURCE_HTTP, restricted),
    m_req(req),
    m_res(req.id()),
    m_url(req.url.c_str())
{
    if (method() == METHOD_GET) {
        if (url() == "/1/summary" || url() == "/2/summary" || url() == "/api.json") {
            m_type = REQ_SUMMARY;
        }
    }

    if (method() == METHOD_POST && url() == "/json_rpc") {
        m_type = REQ_JSON_RPC;
        accept();

        if (hasParseError()) {
            done(RPC_PARSE_ERROR);

            return;
        }

        m_rpcMethod = Json::getString(json(), "method");
        if (m_rpcMethod.isEmpty()) {
            done(RPC_INVALID_REQUEST);

            return;
        }

        m_state = STATE_NEW;

        return;
    }

    if (url().size() > 4) {
        if (memcmp(url().data(), "/2/", 3) == 0) {
            m_version = 2;
        }
    }
}


bool xmrig::HttpApiRequest::accept()
{
    using namespace rapidjson;

    ApiRequest::accept();

    if (m_parsed == 0 && !m_req.body.empty()) {
        m_body.Parse<kParseCommentsFlag | kParseTrailingCommasFlag>(m_req.body.c_str());
        m_parsed = m_body.HasParseError() ? 2 : 1;

        if (!hasParseError()) {
            return true;
        }

        if (type() != REQ_JSON_RPC) {
            reply().AddMember(StringRef(kError), StringRef(GetParseError_En(m_body.GetParseError())), doc().GetAllocator());
        }

        return false;
    }

    return hasParseError();
}


const rapidjson::Value &xmrig::HttpApiRequest::json() const
{
    return m_body;
}


xmrig::IApiRequest::Method xmrig::HttpApiRequest::method() const
{
    return static_cast<IApiRequest::Method>(m_req.method);
}


void xmrig::HttpApiRequest::done(int status)
{
    ApiRequest::done(status);

    if (type() == REQ_JSON_RPC) {
        using namespace rapidjson;
        auto &allocator = doc().GetAllocator();

        m_res.setStatus(HTTP_STATUS_OK);

        if (status != HTTP_STATUS_OK) {
            if (status == HTTP_STATUS_NOT_FOUND) {
                status = RPC_METHOD_NOT_FOUND;
            }

            Value error(kObjectType);
            error.AddMember("code",    status, allocator);
            error.AddMember("message", StringRef(rpcError(status)), allocator);

            reply().AddMember(StringRef(kError), error, allocator);
        }
        else if (!reply().HasMember(kResult)) {
            Value result(kObjectType);
            result.AddMember("status", "OK", allocator);

            reply().AddMember(StringRef(kResult), result, allocator);
        }

        reply().AddMember("jsonrpc", "2.0", allocator);
        reply().AddMember(StringRef(kId), Value().CopyFrom(Json::getValue(json(), kId), allocator), allocator);
    }
    else {
        m_res.setStatus(status);
    }

    m_res.end();
}
