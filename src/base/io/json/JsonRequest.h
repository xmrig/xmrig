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

#ifndef XMRIG_JSONREQUEST_H
#define XMRIG_JSONREQUEST_H


#include "3rdparty/rapidjson/fwd.h"


namespace xmrig {


class JsonRequest
{
public:
    static const char *k2_0;
    static const char *kId;
    static const char *kJsonRPC;
    static const char *kMethod;
    static const char *kOK;
    static const char *kParams;
    static const char *kResult;
    static const char *kStatus;

    static const char *kParseError;
    static const char *kInvalidRequest;
    static const char *kMethodNotFound;
    static const char *kInvalidParams;
    static const char *kInternalError;

    constexpr static int kParseErrorCode        = -32700;
    constexpr static int kInvalidRequestCode    = -32600;
    constexpr static int kMethodNotFoundCode    = -32601;
    constexpr static int kInvalidParamsCode     = -32602;
    constexpr static int kInternalErrorCode     = -32603;

    static rapidjson::Document create(const char *method);
    static rapidjson::Document create(int64_t id, const char *method);
    static uint64_t create(rapidjson::Document &doc, const char *method, rapidjson::Value &params);
    static uint64_t create(rapidjson::Document &doc, int64_t id, const char *method, rapidjson::Value &params);
};


} /* namespace xmrig */


#endif /* XMRIG_JSONREQUEST_H */
