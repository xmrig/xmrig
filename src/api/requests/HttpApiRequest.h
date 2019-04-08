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


#ifndef XMRIG_HTTPAPIREQUEST_H
#define XMRIG_HTTPAPIREQUEST_H


#include "api/requests/ApiRequest.h"
#include "base/net/http/HttpApiResponse.h"
#include "base/tools/String.h"


namespace xmrig {


class HttpData;


class HttpApiRequest : public ApiRequest
{
public:
    HttpApiRequest(const HttpData &req, bool restricted);

protected:
    inline rapidjson::Document &doc() override           { return m_res.doc(); }
    inline rapidjson::Value &reply() override            { return m_res.doc(); }
    inline const String &url() const override            { return m_url; }

    const rapidjson::Value &json() const override;
    Method method() const override;
    void accept() override;
    void done(int status) override;

private:
    bool m_parsed;
    const HttpData &m_req;
    HttpApiResponse m_res;
    rapidjson::Document m_body;
    String m_url;
};


} // namespace xmrig


#endif // XMRIG_HTTPAPIREQUEST_H

