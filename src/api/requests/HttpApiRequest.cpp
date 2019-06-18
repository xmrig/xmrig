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


#include "api/requests/HttpApiRequest.h"
#include "base/net/http/HttpData.h"
#include "rapidjson/error/en.h"


xmrig::HttpApiRequest::HttpApiRequest(const HttpData &req, bool restricted) :
    ApiRequest(SOURCE_HTTP, restricted),
    m_parsed(false),
    m_req(req),
    m_res(req.id()),
    m_url(req.url.c_str())
{
}


const rapidjson::Value &xmrig::HttpApiRequest::json() const
{
    return m_body;
}


xmrig::IApiRequest::Method xmrig::HttpApiRequest::method() const
{
    return static_cast<IApiRequest::Method>(m_req.method);
}


void xmrig::HttpApiRequest::accept()
{
    using namespace rapidjson;

    ApiRequest::accept();

    if (!m_parsed && !m_req.body.empty()) {
        m_parsed = true;
        m_body.Parse<kParseCommentsFlag | kParseTrailingCommasFlag>(m_req.body.c_str());

        if (m_body.HasParseError()) {
            reply().AddMember("error", StringRef(GetParseError_En(m_body.GetParseError())), doc().GetAllocator());;
        }
    }
}


void xmrig::HttpApiRequest::done(int status)
{
    ApiRequest::done(status);

    m_res.setStatus(status);
    m_res.end();
}
