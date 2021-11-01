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


#ifndef XMRIG_HTTPAPIREQUEST_H
#define XMRIG_HTTPAPIREQUEST_H


#include "base/api/requests/ApiRequest.h"
#include "base/net/http/HttpApiResponse.h"
#include "base/tools/String.h"


namespace xmrig {


class HttpData;


class HttpApiRequest : public ApiRequest
{
public:
    HttpApiRequest(const HttpData &req, bool restricted);

protected:
    inline bool hasParseError() const override           { return m_parsed == 2; }
    inline const String &url() const override            { return m_url; }
    inline rapidjson::Document &doc() override           { return m_res.doc(); }
    inline rapidjson::Value &reply() override            { return m_res.doc(); }

    bool accept() override;
    const rapidjson::Value &json() const override;
    Method method() const override;
    void done(int status) override;
    void setRpcError(int code, const char *message = nullptr) override;
    void setRpcResult(rapidjson::Value &result) override;

private:
    void rpcDone(const char *key, rapidjson::Value &value);

    const HttpData &m_req;
    HttpApiResponse m_res;
    int m_parsed = 0;
    rapidjson::Document m_body;
    String m_url;
};


} // namespace xmrig


#endif // XMRIG_HTTPAPIREQUEST_H

