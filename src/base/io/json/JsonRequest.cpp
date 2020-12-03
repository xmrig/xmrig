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


#include "base/io/json/JsonRequest.h"
#include "3rdparty/rapidjson/document.h"


namespace xmrig {


const char *JsonRequest::k2_0               = "2.0";
const char *JsonRequest::kId                = "id";
const char *JsonRequest::kJsonRPC           = "jsonrpc";
const char *JsonRequest::kMethod            = "method";
const char *JsonRequest::kOK                = "OK";
const char *JsonRequest::kParams            = "params";
const char *JsonRequest::kResult            = "result";
const char *JsonRequest::kStatus            = "status";

const char *JsonRequest::kParseError        = "parse error";
const char *JsonRequest::kInvalidRequest    = "invalid request";
const char *JsonRequest::kMethodNotFound    = "method not found";
const char *JsonRequest::kInvalidParams     = "invalid params";
const char *JsonRequest::kInternalError     = "internal error";

static uint64_t nextId                      = 0;


} // namespace xmrig


rapidjson::Document xmrig::JsonRequest::create(const char *method)
{
    return create(++nextId, method);
}


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


uint64_t xmrig::JsonRequest::create(rapidjson::Document &doc, const char *method, rapidjson::Value &params)
{
    return create(doc, ++nextId, method, params);
}


uint64_t xmrig::JsonRequest::create(rapidjson::Document &doc, int64_t id, const char *method, rapidjson::Value &params)
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    doc.AddMember(StringRef(kId),      id, allocator);
    doc.AddMember(StringRef(kJsonRPC), StringRef(k2_0), allocator);
    doc.AddMember(StringRef(kMethod),  StringRef(method), allocator);
    doc.AddMember(StringRef(kParams),  params, allocator);

    return id;
}
