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

#ifndef XMRIG_HTTPRESPONSE_H
#define XMRIG_HTTPRESPONSE_H


#include <map>
#include <string>


namespace xmrig {


class HttpResponse
{
public:
    HttpResponse(uint64_t id, int statusCode = 200);

    inline int statusCode() const                                           { return m_statusCode; }
    inline void setHeader(const std::string &key, const std::string &value) { m_headers.insert({ key, value }); }
    inline void setStatus(int code)                                         { m_statusCode = code; }

    bool isAlive() const;
    void end(const char *data = nullptr, size_t size = 0);

private:
    const uint64_t m_id;
    int m_statusCode;
    std::map<const std::string, const std::string> m_headers;
};


} // namespace xmrig


#endif // XMRIG_HTTPRESPONSE_H

