/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include "base/io/json/JsonRequest.h"
#include "3rdparty/rapidjson/document.h"


namespace xmrig {


static const char *k2_0             = "2.0";
static const char *kId              = "id";
static const char *kJsonRPC         = "jsonrpc";
static const char *kMethod          = "method";
const char *JsonRequest::kParams    = "params";


} // namespace xmrig


rapidjson::Document xmrig::JsonRequest::create(int64_t id, const char *method)
{
    using namespace rapidjson;
    Document doc(kObjectType);
    auto &allocator = doc.GetAllocator();

    doc.AddMember(StringRef(kId),      id, allocator);
    doc.AddMember(StringRef(kJsonRPC), StringRef(k2_0), allocator);
    doc.AddMember(StringRef(kMethod),  StringRef(method), allocator);

    return doc;
}


void xmrig::JsonRequest::create(rapidjson::Document &doc, int64_t id, const char *method, rapidjson::Value &params)
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    doc.AddMember(StringRef(kId),      id, allocator);
    doc.AddMember(StringRef(kJsonRPC), StringRef(k2_0), allocator);
    doc.AddMember(StringRef(kMethod),  StringRef(method), allocator);
    doc.AddMember(StringRef(kParams),  params, allocator);
}
