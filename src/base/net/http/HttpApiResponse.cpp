/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2014-2019 heapwolf    <https://github.com/heapwolf>
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


#include "base/net/http/HttpApiResponse.h"
#include "3rdparty/http-parser/http_parser.h"
#include "3rdparty/rapidjson/prettywriter.h"
#include "3rdparty/rapidjson/stringbuffer.h"
#include "base/net/http/HttpData.h"


namespace xmrig {

static const char *kError  = "error";
static const char *kStatus = "status";

} // namespace xmrig


xmrig::HttpApiResponse::HttpApiResponse(uint64_t id) :
    HttpResponse(id),
    m_doc(rapidjson::kObjectType)
{
}


xmrig::HttpApiResponse::HttpApiResponse(uint64_t id, int status) :
    HttpResponse(id),
    m_doc(rapidjson::kObjectType)
{
    setStatus(status);
}


void xmrig::HttpApiResponse::end()
{
    using namespace rapidjson;

    setHeader("Access-Control-Allow-Origin", "*");
    setHeader("Access-Control-Allow-Methods", "GET, PUT, POST, DELETE");
    setHeader("Access-Control-Allow-Headers", "Authorization, Content-Type");

    if (statusCode() >= 400) {
        if (!m_doc.HasMember(kStatus)) {
            m_doc.AddMember(StringRef(kStatus), statusCode(), m_doc.GetAllocator());
        }

        if (!m_doc.HasMember(kError)) {
            m_doc.AddMember(StringRef(kError), StringRef(http_status_str(static_cast<http_status>(statusCode()))), m_doc.GetAllocator());
        }
    }

    if (!m_doc.MemberCount()) {
        return HttpResponse::end();
    }

    setHeader(HttpData::kContentType, HttpData::kApplicationJson);

    StringBuffer buffer(nullptr, 4096);
    PrettyWriter<StringBuffer> writer(buffer);
    writer.SetMaxDecimalPlaces(10);
    writer.SetFormatOptions(kFormatSingleLineArray);

    m_doc.Accept(writer);

    HttpResponse::end(buffer.GetString(), buffer.GetSize());
}
