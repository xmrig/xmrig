/* XMRig
 * Copyright (c) 2014-2019 heapwolf    <https://github.com/heapwolf>
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

#ifndef XMRIG_HTTPAPIRESPONSE_H
#define XMRIG_HTTPAPIRESPONSE_H


#include "3rdparty/rapidjson/document.h"
#include "base/net/http/HttpResponse.h"


namespace xmrig {


class HttpApiResponse : public HttpResponse
{
public:
    HttpApiResponse(uint64_t id);
    HttpApiResponse(uint64_t id, int status);

    inline rapidjson::Document &doc() { return m_doc; }

    void end();

private:
    rapidjson::Document m_doc;
};


} // namespace xmrig


#endif // XMRIG_HTTPAPIRESPONSE_H

